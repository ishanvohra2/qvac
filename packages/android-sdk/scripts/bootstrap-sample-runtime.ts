import fs from 'fs/promises'
import { createReadStream } from 'fs'
import path from 'path'
import { spawn } from 'child_process'
import { createHash } from 'crypto'
import { fileURLToPath } from 'url'

type BootstrapMetadata = {
  repo: string
  tag: string
  assetName: string
  assetUrl: string
  assetSha256: string
}

const BARE_KIT_REPO = 'holepunchto/bare-kit'
const BARE_KIT_TAG = 'v2.3.0'
const BARE_KIT_ASSET = 'prebuilds.zip'
const BARE_KIT_ASSET_SHA256 = 'a386063fa405b0bb4967490e84745075f007f95359c9871c5b7a45c18c2f49e2'
const BARE_KIT_ASSET_URL = `https://github.com/${BARE_KIT_REPO}/releases/download/${BARE_KIT_TAG}/${BARE_KIT_ASSET}`

const scriptDir = fileURLToPath(new URL('.', import.meta.url))
const packageDir = path.resolve(scriptDir, '..')
const sampleAppDir = path.join(packageDir, 'sample-app')
const targetBareKitDir = path.join(sampleAppDir, 'libs', 'bare-kit')
const targetMetadataPath = path.join(targetBareKitDir, '.bootstrap-metadata.json')

const cacheDir = path.join(packageDir, '.cache', 'bare-kit')
const cachedArchivePath = path.join(cacheDir, `${BARE_KIT_TAG}-${BARE_KIT_ASSET}`)
const extractDir = path.join(cacheDir, `extract-${BARE_KIT_TAG}`)
const localArchivePath = path.join(sampleAppDir, BARE_KIT_ASSET)

const requiredRuntimeFiles = [
  'classes.jar',
  'jni/arm64-v8a/libbare-kit.so',
  'jni/arm64-v8a/libc++_shared.so'
]

function isCheckMode(): boolean {
  return process.argv.includes('--check')
}

async function fileExists(filePath: string): Promise<boolean> {
  try {
    await fs.access(filePath)
    return true
  } catch {
    return false
  }
}

function runCommand(command: string, args: string[], cwd: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd, stdio: 'inherit' })
    child.on('error', reject)
    child.on('close', (code) => {
      if (code === 0) {
        resolve()
      } else {
        reject(new Error(`${command} exited with code ${code ?? 'unknown'}`))
      }
    })
  })
}

function normalizeDigest(value: string): string {
  return value.replace(/^sha256:/i, '').toLowerCase()
}

function toPowerShellLiteral(value: string): string {
  return `'${value.replace(/'/g, "''")}'`
}

async function computeSha256(filePath: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const hash = createHash('sha256')
    const stream = createReadStream(filePath)
    stream.on('error', reject)
    stream.on('data', (chunk: Buffer) => {
      hash.update(chunk)
    })
    stream.on('end', () => {
      resolve(hash.digest('hex'))
    })
  })
}

async function verifyArchiveChecksum(archivePath: string): Promise<void> {
  const actual = await computeSha256(archivePath)
  const expected = normalizeDigest(BARE_KIT_ASSET_SHA256)
  if (actual !== expected) {
    throw new Error(
      `Checksum mismatch for ${archivePath}: expected sha256:${expected}, got sha256:${actual}`
    )
  }
  console.log(`Verified sha256:${actual} for ${path.basename(archivePath)}`)
}

async function getArchivePath(): Promise<string> {
  if (await fileExists(localArchivePath)) {
    console.log(`Using local ${BARE_KIT_ASSET}: ${localArchivePath}`)
    return localArchivePath
  }

  if (await fileExists(cachedArchivePath)) {
    console.log(`Using cached ${BARE_KIT_ASSET}: ${cachedArchivePath}`)
    return cachedArchivePath
  }

  console.log(`Downloading ${BARE_KIT_ASSET_URL}`)
  await fs.mkdir(cacheDir, { recursive: true })
  const response = await fetch(BARE_KIT_ASSET_URL)
  if (!response.ok) {
    throw new Error(`Failed download (${response.status} ${response.statusText}): ${BARE_KIT_ASSET_URL}`)
  }
  const bytes = Buffer.from(await response.arrayBuffer())
  await fs.writeFile(cachedArchivePath, bytes)
  console.log(`Downloaded ${bytes.length} bytes to ${cachedArchivePath}`)
  return cachedArchivePath
}

async function extractArchive(archivePath: string): Promise<string> {
  await fs.rm(extractDir, { recursive: true, force: true })
  await fs.mkdir(extractDir, { recursive: true })
  if (process.platform === 'win32') {
    const command =
      `Expand-Archive -LiteralPath ${toPowerShellLiteral(archivePath)} ` +
      `-DestinationPath ${toPowerShellLiteral(extractDir)} -Force`
    await runCommand('powershell', ['-NoProfile', '-Command', command], packageDir)
  } else {
    await runCommand('unzip', ['-oq', archivePath, '-d', extractDir], packageDir)
  }

  const extractedBareKitDir = path.join(extractDir, 'android', 'bare-kit')
  if (!(await fileExists(extractedBareKitDir))) {
    throw new Error(`Expected extracted bare-kit dir not found: ${extractedBareKitDir}`)
  }
  return extractedBareKitDir
}

async function writeMetadata(): Promise<void> {
  const metadata: BootstrapMetadata = {
    repo: BARE_KIT_REPO,
    tag: BARE_KIT_TAG,
    assetName: BARE_KIT_ASSET,
    assetUrl: BARE_KIT_ASSET_URL,
    assetSha256: normalizeDigest(BARE_KIT_ASSET_SHA256)
  }
  await fs.writeFile(targetMetadataPath, `${JSON.stringify(metadata, null, 2)}\n`)
}

async function validateBootstrapState(): Promise<void> {
  for (const relativePath of requiredRuntimeFiles) {
    const fullPath = path.join(targetBareKitDir, relativePath)
    if (!(await fileExists(fullPath))) {
      throw new Error(
        `Missing required Bare Kit runtime file: ${fullPath}\nRun: bun run sample:bootstrap-runtime`
      )
    }
  }

  if (!(await fileExists(targetMetadataPath))) {
    throw new Error(
      `Missing Bare Kit bootstrap metadata: ${targetMetadataPath}\nRun: bun run sample:bootstrap-runtime`
    )
  }

  const metadataRaw = await fs.readFile(targetMetadataPath, 'utf8')
  const metadata = JSON.parse(metadataRaw) as BootstrapMetadata
  const expectedDigest = normalizeDigest(BARE_KIT_ASSET_SHA256)
  if (
    metadata.repo !== BARE_KIT_REPO ||
    metadata.tag !== BARE_KIT_TAG ||
    normalizeDigest(metadata.assetSha256 ?? '') !== expectedDigest
  ) {
    throw new Error(
      `Bare Kit runtime metadata mismatch. Expected ${BARE_KIT_REPO}@${BARE_KIT_TAG} sha256:${expectedDigest}`
    )
  }
}

async function bootstrapRuntime(): Promise<void> {
  const archivePath = await getArchivePath()
  await verifyArchiveChecksum(archivePath)
  const extractedBareKitDir = await extractArchive(archivePath)

  await fs.rm(targetBareKitDir, { recursive: true, force: true })
  await fs.mkdir(path.dirname(targetBareKitDir), { recursive: true })
  await fs.cp(extractedBareKitDir, targetBareKitDir, { recursive: true })
  await writeMetadata()
  await validateBootstrapState()
  console.log(`Bootstrapped Bare Kit runtime at ${targetBareKitDir}`)
}

async function main(): Promise<void> {
  if (isCheckMode()) {
    await validateBootstrapState()
    console.log(`Bare Kit runtime is present and pinned to ${BARE_KIT_REPO}@${BARE_KIT_TAG}`)
    return
  }
  await bootstrapRuntime()
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error))
  process.exitCode = 1
})
