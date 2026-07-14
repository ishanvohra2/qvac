import fs from 'fs/promises'
import path from 'path'
import { fileURLToPath, pathToFileURL } from 'url'

type AddonPackage = {
  packageDir: string
  packageName: string
  packageVersion: string
  bareBinaryPath: string
}

const scriptDir = fileURLToPath(new URL('.', import.meta.url))
const packageDir = path.resolve(scriptDir, '..')
const nodeModulesRoot = path.join(packageDir, 'node_modules')
const outputDir = path.join(packageDir, 'sample-app', 'src', 'main', 'addons', 'arm64-v8a')

async function pathExists(targetPath: string): Promise<boolean> {
  try {
    await fs.access(targetPath)
    return true
  } catch {
    return false
  }
}

async function listDirs(targetPath: string): Promise<string[]> {
  const entries = await fs.readdir(targetPath, { withFileTypes: true })
  return entries.filter((entry) => entry.isDirectory()).map((entry) => path.join(targetPath, entry.name))
}

async function findPackageJsonFiles(root: string): Promise<string[]> {
  const found: string[] = []
  const queue: string[] = [root]

  while (queue.length > 0) {
    const current = queue.shift()
    if (!current) continue

    const childDirs = await listDirs(current)
    for (const childDir of childDirs) {
      const base = path.basename(childDir)
      if (base.startsWith('.')) continue
      if (base === '.bin') continue
      if (base === 'prebuilds') continue
      queue.push(childDir)
    }

    const packageJsonPath = path.join(current, 'package.json')
    if (await pathExists(packageJsonPath)) {
      found.push(packageJsonPath)
    }
  }

  return found
}

export function normalizeSharedObjectBaseName(packageName: string): string {
  if (packageName.startsWith('@')) {
    return packageName.slice(1).replace('/', '__')
  }
  return packageName
}

export function isBarePrebuild(entry: { isFile(): boolean; name: string }): boolean {
  return entry.isFile() && entry.name.endsWith('.bare')
}

async function discoverAddonPackages(): Promise<AddonPackage[]> {
  const packageJsonFiles = await findPackageJsonFiles(nodeModulesRoot)
  const addons: AddonPackage[] = []

  for (const packageJsonPath of packageJsonFiles) {
    const packageDirPath = path.dirname(packageJsonPath)
    const packageJsonRaw = await fs.readFile(packageJsonPath, 'utf8')
    const packageJson = JSON.parse(packageJsonRaw) as {
      name?: string
      version?: string
      addon?: boolean
    }

    if (packageJson.addon !== true) continue
    if (!packageJson.name || !packageJson.version) continue

    const prebuildDir = path.join(packageDirPath, 'prebuilds', 'android-arm64')
    if (!(await pathExists(prebuildDir))) continue

    const prebuildEntries = await fs.readdir(prebuildDir, { withFileTypes: true })
    const bareBinary = prebuildEntries.find(isBarePrebuild)
    if (!bareBinary) continue

    addons.push({
      packageDir: packageDirPath,
      packageName: packageJson.name,
      packageVersion: packageJson.version,
      bareBinaryPath: path.join(prebuildDir, bareBinary.name)
    })
  }

  return addons.sort((a, b) => a.packageName.localeCompare(b.packageName))
}

async function main(): Promise<void> {
  if (!(await pathExists(nodeModulesRoot))) {
    throw new Error(`Missing node_modules at ${nodeModulesRoot}`)
  }

  const addonPackages = await discoverAddonPackages()
  if (addonPackages.length === 0) {
    throw new Error('No addon packages with android-arm64 .bare prebuilds were found')
  }

  await fs.mkdir(outputDir, { recursive: true })

  for (const addon of addonPackages) {
    const baseName = normalizeSharedObjectBaseName(addon.packageName)
    const destination = path.join(outputDir, `lib${baseName}.${addon.packageVersion}.so`)
    await fs.copyFile(addon.bareBinaryPath, destination)
  }

  console.log(
    `Discovered and copied ${addonPackages.length} runtime addons to ${outputDir}`
  )
}

const isMainModule = import.meta.url === pathToFileURL(process.argv[1] ?? '').href

if (isMainModule) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error))
    process.exitCode = 1
  })
}
