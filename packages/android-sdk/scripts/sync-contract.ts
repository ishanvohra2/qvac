import fs from 'fs/promises'
import path from 'path'
import { fileURLToPath } from 'url'
import { spawn } from 'child_process'

type SyncEntry = {
  sourceRelativePath: string
  destinationRelativePath: string
}

type SdkManifest = {
  sdk: {
    version: string
  }
}

type PackageJson = {
  version: string
}

const scriptDir = fileURLToPath(new URL('.', import.meta.url))
const packageDir = path.resolve(scriptDir, '..')
const sdkGeneratedDir = path.resolve(packageDir, '../sdk/android/generated')
const packageJsonPath = path.join(packageDir, 'package.json')
const sdkManifestPath = path.join(sdkGeneratedDir, 'qvac-sdk-manifest.json')
const syncEntriesPath = path.join(packageDir, 'scripts', 'sync-contract-entries.json')
const generateBindingsScriptPath = path.join(packageDir, 'scripts', 'generate-sample-bindings.mjs')

async function readFileSafe(filePath: string): Promise<string | null> {
  try {
    return await fs.readFile(filePath, 'utf8')
  } catch {
    return null
  }
}

async function ensureParentDirectory(filePath: string): Promise<void> {
  const directory = path.dirname(filePath)
  await fs.mkdir(directory, { recursive: true })
}

async function syncEntry(entry: SyncEntry, checkOnly: boolean): Promise<boolean> {
  const sourcePath = path.join(sdkGeneratedDir, entry.sourceRelativePath)
  const destinationPath = path.join(packageDir, entry.destinationRelativePath)
  const sourceContent = await readFileSafe(sourcePath)

  if (sourceContent === null) {
    throw new Error(`Missing source contract file: ${sourcePath}`)
  }

  const destinationContent = await readFileSafe(destinationPath)
  if (destinationContent === sourceContent) {
    console.log(`UNCHANGED ${entry.destinationRelativePath}`)
    return false
  }

  if (checkOnly) {
    console.log(`OUTDATED ${entry.destinationRelativePath}`)
    return true
  }

  await ensureParentDirectory(destinationPath)
  await fs.writeFile(destinationPath, sourceContent)
  console.log(`UPDATED ${entry.destinationRelativePath}`)
  return true
}

async function syncPackageVersion(checkOnly: boolean): Promise<boolean> {
  const sdkManifestRaw = await readFileSafe(sdkManifestPath)
  if (sdkManifestRaw === null) {
    throw new Error(`Missing source contract file: ${sdkManifestPath}`)
  }
  const sdkManifest = JSON.parse(sdkManifestRaw) as SdkManifest
  const packageJsonRaw = await readFileSafe(packageJsonPath)
  if (packageJsonRaw === null) {
    throw new Error(`Missing package.json: ${packageJsonPath}`)
  }
  const packageJson = JSON.parse(packageJsonRaw) as PackageJson
  if (packageJson.version === sdkManifest.sdk.version) {
    console.log('UNCHANGED package.json version')
    return false
  }

  if (checkOnly) {
    console.log(`OUTDATED package.json version (expected ${sdkManifest.sdk.version})`)
    return true
  }

  const nextPackageJson = {
    ...JSON.parse(packageJsonRaw),
    version: sdkManifest.sdk.version
  }
  await fs.writeFile(packageJsonPath, `${JSON.stringify(nextPackageJson, null, 2)}\n`)
  console.log(`UPDATED package.json version -> ${sdkManifest.sdk.version}`)
  return true
}

async function runBindingsGenerator(checkOnly: boolean): Promise<boolean> {
  const args = [generateBindingsScriptPath]
  if (checkOnly) {
    args.push('--check')
  }
  return new Promise((resolve, reject) => {
    const child = spawn('node', args, { cwd: packageDir, stdio: 'inherit' })
    child.on('error', reject)
    child.on('close', (code) => {
      if (code === 0) {
        resolve(false)
        return
      }
      // In --check mode exit code 1 signals drift (files need regenerating), not a
      // failure. In write mode the generator only exits non-zero on a real error, so
      // any non-zero code must surface instead of being counted as a synced file.
      if (checkOnly && code === 1) {
        resolve(true)
        return
      }
      reject(new Error(`Bindings generator exited with code ${code ?? 'unknown'}`))
    })
  })
}

async function main(): Promise<void> {
  const checkOnly = process.argv.includes('--check')
  let changedCount = 0
  const syncEntriesRaw = await readFileSafe(syncEntriesPath)
  if (syncEntriesRaw === null) {
    throw new Error(`Missing sync entries file: ${syncEntriesPath}`)
  }
  const syncEntries = JSON.parse(syncEntriesRaw) as SyncEntry[]

  for (const entry of syncEntries) {
    const changed = await syncEntry(entry, checkOnly)
    if (changed) {
      changedCount += 1
    }
  }
  const versionChanged = await syncPackageVersion(checkOnly)
  if (versionChanged) {
    changedCount += 1
  }
  const bindingsChanged = await runBindingsGenerator(checkOnly)
  if (bindingsChanged) {
    changedCount += 1
  }

  if (checkOnly) {
    if (changedCount > 0) {
      process.exitCode = 1
      return
    }
    console.log('Android SDK contract files are up to date')
    return
  }

  console.log(
    `android:sync-contract complete (${changedCount} file${changedCount === 1 ? '' : 's'} changed)`
  )
}

main().catch((error) => {
  console.error(error)
  process.exitCode = 1
})
