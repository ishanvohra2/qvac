import fs from 'fs/promises'
import path from 'path'
import { fileURLToPath } from 'url'
import { models } from '@/models/registry/models'
import {
  collectApiOperationsFromSources,
  collectDependencies,
  toCamelCase
} from './generate-utils'
import {
  androidManifestSourceSchema,
  generatedAndroidManifestSchema,
  generatedModelConstantSchema,
  type AndroidManifestSource,
  type GeneratedAddonCapability,
  type GeneratedApiOperation,
  type GeneratedAndroidManifest,
  type GeneratedDependency,
  type GeneratedModelConstant
} from './types'

type PackageJson = {
  name: string
  version: string
  dependencies?: Record<string, string>
  peerDependencies?: Record<string, string>
}

const scriptDir = fileURLToPath(new URL('.', import.meta.url))
const sdkDir = path.resolve(scriptDir, '../..')
const sourceManifestPath = path.join(sdkDir, 'android', 'manifest.source.json')
const packageJsonPath = path.join(sdkDir, 'package.json')
const generatedDir = path.join(sdkDir, 'android', 'generated')
const schemasDir = path.join(sdkDir, 'schemas')

const outputFiles = {
  manifest: path.join(generatedDir, 'qvac-sdk-manifest.json'),
  capabilities: path.join(generatedDir, 'capabilities.json'),
  modelConstants: path.join(generatedDir, 'models-catalog.json'),
  apiContract: path.join(generatedDir, 'api-contract.json'),
  gradleVersionCatalog: path.join(generatedDir, 'libs.versions.toml'),
  kotlinInfo: path.join(generatedDir, 'GeneratedQvacSdkInfo.kt'),
  kotlinApi: path.join(generatedDir, 'GeneratedQvacApi.kt'),
  addonManifest: path.join(generatedDir, 'addon-manifest.json')
} as const

async function readJsonFile<T>(filePath: string): Promise<T> {
  const content = await fs.readFile(filePath, 'utf8')
  return JSON.parse(content) as T
}

function collectCapabilities(source: AndroidManifestSource): GeneratedAddonCapability[] {
  type AddonPolicyMap = AndroidManifestSource['addonPolicy']
  type AddonPolicyKey = Extract<keyof AddonPolicyMap, string>

  const engineSetByAddon = new Map<string, Set<string>>()
  const countByAddon = new Map<string, number>()

  for (const model of models) {
    const addon = model.addon
    const engine = model.engine
    const existingSet = engineSetByAddon.get(addon) ?? new Set<string>()
    existingSet.add(engine)
    engineSetByAddon.set(addon, existingSet)
    const currentCount = countByAddon.get(addon) ?? 0
    countByAddon.set(addon, currentCount + 1)
  }

  const capabilities: GeneratedAddonCapability[] = []
  const addonKeys = (Object.keys(source.addonPolicy) as AddonPolicyKey[]).sort((a, b) =>
    a.localeCompare(b)
  )
  for (const addon of addonKeys) {
    const policy = source.addonPolicy[addon]
    if (!policy) continue
    const engines = Array.from(engineSetByAddon.get(addon) ?? []).sort((a, b) => a.localeCompare(b))
    capabilities.push({
      addon,
      androidSupported: policy.androidSupported,
      fallbackBehavior: policy.fallbackBehavior,
      engines,
      modelCount: countByAddon.get(addon) ?? 0
    })
  }

  return capabilities
}

function createGeneratedManifest(
  packageJson: PackageJson,
  source: AndroidManifestSource,
  dependencies: GeneratedDependency[],
  capabilities: GeneratedAddonCapability[]
): GeneratedAndroidManifest {
  return {
    schemaVersion: 1,
    sourceSchemaVersion: source.schemaVersion,
    generatedAt: `sdk-version-${packageJson.version}`,
    sdk: {
      packageName: '@qvac/sdk',
      version: packageJson.version
    },
    android: source.android,
    runtime: source.runtime,
    dependencies,
    capabilities
  }
}

function collectModelConstants(): GeneratedModelConstant[] {
  const constants: GeneratedModelConstant[] = []
  for (const model of models) {
    const parsed = generatedModelConstantSchema.parse({
      name: model.name,
      src: `registry://${model.registrySource}/${model.registryPath}`,
      modelId: model.modelId,
      registryPath: model.registryPath,
      registrySource: model.registrySource,
      addon: model.addon,
      engine: model.engine,
      quantization: model.quantization,
      params: model.params
    })
    constants.push(parsed)
  }

  constants.sort((a, b) => a.name.localeCompare(b.name))
  return constants
}

async function collectApiOperations(): Promise<GeneratedApiOperation[]> {
  const fileNames = (await fs.readdir(schemasDir)).filter((name) => name.endsWith('.ts')).sort()
  const sourceFiles: Array<{ fileName: string; content: string }> = []
  for (const fileName of fileNames) {
    const sourceFile = path.join(schemasDir, fileName)
    const content = await fs.readFile(sourceFile, 'utf8')
    sourceFiles.push({
      fileName,
      content
    })
  }

  return collectApiOperationsFromSources(sourceFiles)
}

function toKotlinApi(operations: GeneratedApiOperation[]): string {
  const kotlinReservedNames = new Set([
    'as', 'break', 'class', 'continue', 'do', 'else', 'false', 'for', 'fun', 'if', 'in',
    'interface', 'is', 'null', 'object', 'package', 'return', 'super', 'this', 'throw',
    'true', 'try', 'typealias', 'typeof', 'val', 'var', 'when', 'while', 'by', 'catch',
    'constructor', 'delegate', 'dynamic', 'field', 'file', 'finally', 'get', 'import',
    'init', 'param', 'property', 'receiver', 'set', 'setparam', 'where', 'actual',
    'abstract', 'annotation', 'companion', 'const', 'crossinline', 'data', 'enum',
    'expect', 'external', 'final', 'infix', 'inline', 'inner', 'internal', 'lateinit',
    'noinline', 'open', 'operator', 'out', 'override', 'private', 'protected', 'public',
    'reified', 'sealed', 'suspend', 'tailrec', 'vararg', 'yield'
  ])
  function toKotlinFunctionName(operation: string): string {
    const candidate = toCamelCase(operation)
    if (!kotlinReservedNames.has(candidate)) return candidate
    return `${candidate}Operation`
  }

  const lines: string[] = []
  lines.push('// AUTO-GENERATED BY scripts/android/generate.ts')
  lines.push('// DO NOT MODIFY MANUALLY')
  lines.push('')
  lines.push('package io.tether.qvac.sdk.generated.api')
  lines.push('')
  lines.push('import kotlinx.coroutines.flow.Flow')
  lines.push('import org.json.JSONObject')
  lines.push('')

  for (const operation of operations) {
    lines.push(`data class ${operation.requestTypeName}(val payload: JSONObject = JSONObject())`)
    if (operation.streaming) {
      lines.push(`data class ${operation.responseTypeName}(val payload: JSONObject = JSONObject())`)
    } else {
      lines.push(`data class ${operation.responseTypeName}(val payload: JSONObject = JSONObject())`)
    }
    lines.push('')
  }

  lines.push('// NOTE: These are generated schema wrappers only.')
  lines.push('// The sample app still uses a separate ad-hoc IPC protocol.')
  lines.push('interface QvacGeneratedApiClient {')
  for (const operation of operations) {
    const functionName = toKotlinFunctionName(operation.operation)
    if (operation.streaming) {
      lines.push(
        `  fun ${functionName}(request: ${operation.requestTypeName}): Flow<${operation.responseTypeName}>`
      )
    } else {
      lines.push(
        `  suspend fun ${functionName}(request: ${operation.requestTypeName}): ${operation.responseTypeName}`
      )
    }
  }
  lines.push('}')
  lines.push('')
  lines.push('object QvacGeneratedApiContract {')
  lines.push('  val operations: List<String> = listOf(')
  for (const operation of operations) {
    lines.push(`    "${operation.operation}",`)
  }
  lines.push('  )')
  lines.push('}')
  lines.push('')

  return `${lines.join('\n')}\n`
}

function sanitizeTomlKey(value: string): string {
  return value.replace(/[^a-zA-Z0-9]/g, '_')
}

function toGradleVersionCatalog(
  dependencies: GeneratedDependency[],
  manifest: GeneratedAndroidManifest
): string {
  const lines: string[] = []
  lines.push('# AUTO-GENERATED BY scripts/android/generate.ts')
  lines.push('# DO NOT MODIFY MANUALLY')
  lines.push('')
  lines.push('[versions]')
  lines.push(`qvac_sdk = "${manifest.sdk.version}"`)
  for (const dependency of dependencies) {
    lines.push(`${sanitizeTomlKey(dependency.packageName)} = "${dependency.version}"`)
  }
  lines.push('')
  lines.push('[libraries]')
  for (const dependency of dependencies) {
    const key = sanitizeTomlKey(dependency.packageName)
    lines.push(`${key} = { module = "${dependency.packageName}", version.ref = "${key}" }`)
  }
  lines.push('')
  return `${lines.join('\n')}\n`
}

function toKotlinInfo(manifest: GeneratedAndroidManifest): string {
  const abiList = manifest.android.abis.join(', ')
  return `// AUTO-GENERATED BY scripts/android/generate.ts
// DO NOT MODIFY MANUALLY

package ${manifest.android.kotlinPackage}

object GeneratedQvacSdkInfo {
  const val VERSION = "${manifest.sdk.version}"
  const val GROUP_ID = "${manifest.android.groupId}"
  const val ARTIFACT_ID = "${manifest.android.artifactId}"
  const val NAMESPACE = "${manifest.android.namespace}"
  const val MIN_SDK = ${manifest.android.minSdk}
  const val TARGET_SDK = ${manifest.android.targetSdk}
  const val COMPILE_SDK = ${manifest.android.compileSdk}
  const val ABIS = "${abiList}"
}
`
}

function toAddonManifest(capabilities: GeneratedAddonCapability[]): string {
  const addonManifest = {
    schemaVersion: 1,
    addons: capabilities.map((capability) => ({
      addon: capability.addon,
      androidSupported: capability.androidSupported,
      fallbackBehavior: capability.fallbackBehavior
    }))
  }
  return `${JSON.stringify(addonManifest, null, 2)}\n`
}

async function readIfExists(filePath: string): Promise<string | null> {
  try {
    return await fs.readFile(filePath, 'utf8')
  } catch {
    return null
  }
}

async function ensureDirectory(directoryPath: string): Promise<void> {
  await fs.mkdir(directoryPath, { recursive: true })
}

async function writeFileIfChanged(filePath: string, content: string): Promise<boolean> {
  const previous = await readIfExists(filePath)
  if (previous === content) return false
  await fs.writeFile(filePath, content)
  return true
}

async function checkFileMatches(filePath: string, expected: string): Promise<boolean> {
  const current = await readIfExists(filePath)
  return current === expected
}

async function main(): Promise<void> {
  const checkMode = process.argv.includes('--check')
  const sourceRaw = await readJsonFile<unknown>(sourceManifestPath)
  const source = androidManifestSourceSchema.parse(sourceRaw)
  const packageJson = await readJsonFile<PackageJson>(packageJsonPath)

  if (packageJson.name !== '@qvac/sdk') {
    throw new Error(`Expected package name @qvac/sdk, got ${packageJson.name}`)
  }

  const dependencies = collectDependencies(packageJson, source)
  const capabilities = collectCapabilities(source)
  const generatedManifest = generatedAndroidManifestSchema.parse(
    createGeneratedManifest(packageJson, source, dependencies, capabilities)
  )
  const modelConstants = collectModelConstants()
  const apiOperations = await collectApiOperations()

  const manifestJson = `${JSON.stringify(generatedManifest, null, 2)}\n`
  const capabilitiesJson = `${JSON.stringify(generatedManifest.capabilities, null, 2)}\n`
  const modelConstantsJson = `${JSON.stringify(modelConstants, null, 2)}\n`
  const apiContractJson = `${JSON.stringify(apiOperations, null, 2)}\n`
  const gradleToml = toGradleVersionCatalog(generatedManifest.dependencies, generatedManifest)
  const kotlinInfo = toKotlinInfo(generatedManifest)
  const kotlinApi = toKotlinApi(apiOperations)
  const addonManifest = toAddonManifest(generatedManifest.capabilities)

  const outputs: Record<string, string> = {
    [outputFiles.manifest]: manifestJson,
    [outputFiles.capabilities]: capabilitiesJson,
    [outputFiles.modelConstants]: modelConstantsJson,
    [outputFiles.apiContract]: apiContractJson,
    [outputFiles.gradleVersionCatalog]: gradleToml,
    [outputFiles.kotlinInfo]: kotlinInfo,
    [outputFiles.kotlinApi]: kotlinApi,
    [outputFiles.addonManifest]: addonManifest
  }

  if (checkMode) {
    let hasDrift = false
    const outputPaths = Object.keys(outputs).sort((a, b) => a.localeCompare(b))
    for (const outputPath of outputPaths) {
      const matches = await checkFileMatches(outputPath, outputs[outputPath]!)
      if (!matches) {
        hasDrift = true
        console.log(`OUTDATED ${path.relative(sdkDir, outputPath)}`)
      }
    }
    if (hasDrift) {
      process.exitCode = 1
      return
    }
    console.log('Android generated files are up to date')
    return
  }

  await ensureDirectory(generatedDir)
  const outputPaths = Object.keys(outputs).sort((a, b) => a.localeCompare(b))
  let changed = 0
  for (const outputPath of outputPaths) {
    const didWrite = await writeFileIfChanged(outputPath, outputs[outputPath]!)
    if (didWrite) {
      changed += 1
      console.log(`UPDATED ${path.relative(sdkDir, outputPath)}`)
    } else {
      console.log(`UNCHANGED ${path.relative(sdkDir, outputPath)}`)
    }
  }
  console.log(`android:sync complete (${changed} file${changed === 1 ? '' : 's'} changed)`)
}

main().catch((error) => {
  console.error(error)
  process.exitCode = 1
})
