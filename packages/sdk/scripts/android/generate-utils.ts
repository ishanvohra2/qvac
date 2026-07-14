import type {
  AndroidManifestSource,
  GeneratedAddonCapability,
  GeneratedAndroidManifest,
  GeneratedApiOperation,
  GeneratedDependency,
  GeneratedModelConstant
} from './types'
import { generatedApiOperationSchema, generatedModelConstantSchema } from './types'

export type CapabilityModel = {
  addon: string
  engine: string
}

export type ModelConstantInput = {
  name: string
  registrySource: string
  registryPath: string
  modelId: string
  addon: string
  engine: string
  quantization: string
  params: string
}

type PackageJson = {
  dependencies?: Record<string, string>
  peerDependencies?: Record<string, string>
}

type SchemaSourceFile = {
  fileName: string
  content: string
}

export function shouldIncludeDependency(packageName: string, source: AndroidManifestSource): boolean {
  const includeByPrefix = source.dependencyPolicy.includePrefixes.some((prefix) =>
    packageName.startsWith(prefix)
  )
  if (!includeByPrefix) return false
  return !source.dependencyPolicy.excludePackages.includes(packageName)
}

function getScopeDependencies(
  packageJson: PackageJson,
  scopeName: 'dependencies' | 'peerDependencies'
): Record<string, string> {
  if (scopeName === 'dependencies') return packageJson.dependencies ?? {}
  return packageJson.peerDependencies ?? {}
}

function collectScopeDependencies(
  packageJson: PackageJson,
  source: AndroidManifestSource,
  scopeName: 'dependencies' | 'peerDependencies'
): GeneratedDependency[] {
  const scoped = getScopeDependencies(packageJson, scopeName)
  const dependencies: GeneratedDependency[] = []
  for (const packageName of Object.keys(scoped)) {
    if (!shouldIncludeDependency(packageName, source)) continue
    dependencies.push({
      packageName,
      version: scoped[packageName]!,
      sourceScope: scopeName
    })
  }
  return dependencies
}

export function collectDependencies(
  packageJson: PackageJson,
  source: AndroidManifestSource
): GeneratedDependency[] {
  const dependencies: GeneratedDependency[] = []

  for (const scopeName of source.dependencyPolicy.includeScopes) {
    dependencies.push(...collectScopeDependencies(packageJson, source, scopeName))
  }

  dependencies.sort(
    (a, b) =>
      a.packageName.localeCompare(b.packageName) || a.sourceScope.localeCompare(b.sourceScope)
  )

  return dependencies
}

export function toPascalCase(value: string): string {
  return value
    .replace(/[^a-zA-Z0-9]+/g, ' ')
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join('')
}

export function toCamelCase(value: string): string {
  const pascal = toPascalCase(value)
  return pascal.length === 0 ? pascal : pascal.charAt(0).toLowerCase() + pascal.slice(1)
}

function collectSchemaNames(content: string): string[] {
  const names: string[] = []
  const pattern = /export const\s+([A-Za-z0-9_]+(?:Request|Response)Schema)\s*=/g
  let match: RegExpExecArray | null
  while ((match = pattern.exec(content)) !== null) {
    names.push(match[1]!)
  }
  return names
}

function readDeclarationBlock(content: string, schemaName: string): string | null {
  const escapedSchemaName = schemaName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const declarationRegex = new RegExp(
    `export const\\s+${escapedSchemaName}\\s*=([\\s\\S]*?)(?=\\nexport const\\s+\\w+\\s*=|\\nexport type\\s+\\w+\\s*=|\\nexport interface\\s+\\w+\\s*|\\n$|$)`
  )
  const match = declarationRegex.exec(content)
  if (!match) return null
  return match[1] ?? null
}

// Extracts the discriminant from the FIRST `type:` field of a schema block. It is
// deliberately strict: the field must be `type: z.literal('...')` (whitespace/newlines
// between tokens are fine, trailing `.describe(...)` etc. is ignored). If the first
// `type:` is a `z.enum(...)`, a computed value, or otherwise not an inline literal, we
// return null and let the registry fallback decide — rather than lazily scanning ahead
// and silently binding the operation to an unrelated nested `z.literal`.
function readOperationLiteral(schemaBlock: string): string | null {
  const firstTypeField = /\btype\s*:\s*/.exec(schemaBlock)
  if (!firstTypeField) return null
  const afterType = schemaBlock.slice(firstTypeField.index + firstTypeField[0].length)
  const literal = /^z\s*\.\s*literal\s*\(\s*(['"`])([^'"`]+)\1\s*\)/.exec(afterType)
  if (!literal) return null
  return literal[2] ?? null
}

const registryOperationPattern = /^\s*([A-Za-z0-9_]+):\s*\{\s*type:\s*'([^']+)'/gm

function parseRegistryOperations(registrySource: string): Array<{ operation: string; streaming: boolean }> {
  const parsed: Array<{ operation: string; streaming: boolean }> = []
  const pattern = new RegExp(registryOperationPattern.source, registryOperationPattern.flags)
  let match: RegExpExecArray | null
  while ((match = pattern.exec(registrySource)) !== null) {
    const operation = match[1]!
    const handlerType = match[2]!
    parsed.push({
      operation,
      streaming: handlerType === 'stream' || handlerType === 'duplex'
    })
  }
  return parsed
}

function buildRegistryOperation(operation: string, streaming: boolean): GeneratedApiOperation {
  const baseName = toPascalCase(operation)
  return generatedApiOperationSchema.parse({
    operation,
    requestSchema: `${toCamelCase(operation)}RequestSchema`,
    responseSchema: streaming
      ? `${toCamelCase(operation)}StreamResponseSchema`
      : `${toCamelCase(operation)}ResponseSchema`,
    requestTypeName: `${baseName}Request`,
    responseTypeName: `${baseName}${streaming ? 'StreamEvent' : 'Response'}`,
    streaming,
    sourceFile: 'handler-registry.ts'
  })
}

export function mergeRegistryOperations(
  fromSchemas: GeneratedApiOperation[],
  registrySource: string
): GeneratedApiOperation[] {
  const byOperation = new Map(fromSchemas.map((entry) => [entry.operation, entry]))
  for (const { operation, streaming } of parseRegistryOperations(registrySource)) {
    if (byOperation.has(operation)) continue
    byOperation.set(operation, buildRegistryOperation(operation, streaming))
  }
  return Array.from(byOperation.values()).sort((a, b) => a.operation.localeCompare(b.operation))
}

export function collectApiOperationsFromSources(
  sourceFiles: SchemaSourceFile[]
): GeneratedApiOperation[] {
  const requestByOperation = new Map<
    string,
    { requestSchema: string; sourceFile: string }
  >()
  const responseByOperation = new Map<string, string>()

  for (const sourceFile of sourceFiles) {
    const schemaNames = collectSchemaNames(sourceFile.content)
    for (const schemaName of schemaNames) {
      const declarationBlock = readDeclarationBlock(sourceFile.content, schemaName)
      if (declarationBlock === null) continue
      const operation = readOperationLiteral(declarationBlock)
      if (operation === null) continue

      if (schemaName.endsWith('RequestSchema')) {
        requestByOperation.set(operation, {
          requestSchema: schemaName,
          sourceFile: sourceFile.fileName
        })
      }
      if (schemaName.endsWith('ResponseSchema')) {
        responseByOperation.set(operation, schemaName)
      }
    }
  }

  const operations: GeneratedApiOperation[] = []
  for (const [operation, requestInfo] of requestByOperation.entries()) {
    const baseName = toPascalCase(operation)
    const responseSchema = responseByOperation.get(operation) ?? null
    const streaming =
      requestInfo.requestSchema.endsWith('StreamRequestSchema') ||
      (responseSchema?.endsWith('StreamResponseSchema') ?? false)
    const parsed = generatedApiOperationSchema.parse({
      operation,
      requestSchema: requestInfo.requestSchema,
      responseSchema,
      requestTypeName: `${baseName}Request`,
      responseTypeName: `${baseName}${streaming ? 'StreamEvent' : 'Response'}`,
      streaming,
      sourceFile: requestInfo.sourceFile
    })
    operations.push(parsed)
  }

  operations.sort((a, b) => a.operation.localeCompare(b.operation))
  return operations
}

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

type AddonModelStats = {
  engineSetByAddon: Map<string, Set<string>>
  countByAddon: Map<string, number>
}

export function collectAddonModelStats(models: CapabilityModel[]): AddonModelStats {
  const engineSetByAddon = new Map<string, Set<string>>()
  const countByAddon = new Map<string, number>()

  for (const model of models) {
    const existingSet = engineSetByAddon.get(model.addon) ?? new Set<string>()
    existingSet.add(model.engine)
    engineSetByAddon.set(model.addon, existingSet)
    countByAddon.set(model.addon, (countByAddon.get(model.addon) ?? 0) + 1)
  }

  return { engineSetByAddon, countByAddon }
}

export function buildCapabilities(
  source: AndroidManifestSource,
  models: CapabilityModel[]
): GeneratedAddonCapability[] {
  type AddonPolicyKey = Extract<keyof AndroidManifestSource['addonPolicy'], string>

  const { engineSetByAddon, countByAddon } = collectAddonModelStats(models)
  const addonKeys = (Object.keys(source.addonPolicy) as AddonPolicyKey[]).sort((a, b) =>
    a.localeCompare(b)
  )

  const capabilities: GeneratedAddonCapability[] = []
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

export function buildModelConstants(models: ModelConstantInput[]): GeneratedModelConstant[] {
  const constants = models.map((model) =>
    generatedModelConstantSchema.parse({
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
  )
  constants.sort((a, b) => a.name.localeCompare(b.name))
  return constants
}

export function toKotlinFunctionName(operation: string): string {
  const candidate = toCamelCase(operation)
  if (!kotlinReservedNames.has(candidate)) return candidate
  return `${candidate}Operation`
}

function renderKotlinDataClasses(operations: GeneratedApiOperation[]): string[] {
  const lines: string[] = []
  for (const operation of operations) {
    lines.push(`data class ${operation.requestTypeName}(val payload: JSONObject = JSONObject())`)
    lines.push(`data class ${operation.responseTypeName}(val payload: JSONObject = JSONObject())`)
    lines.push('')
  }
  return lines
}

function renderKotlinInterfaceMethods(operations: GeneratedApiOperation[]): string[] {
  const lines: string[] = []
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
  return lines
}

function renderKotlinOperationList(operations: GeneratedApiOperation[]): string[] {
  return operations.map((operation) => `    "${operation.operation}",`)
}

export function toKotlinApi(operations: GeneratedApiOperation[]): string {
  const lines: string[] = []
  lines.push('// AUTO-GENERATED BY scripts/android/generate.ts')
  lines.push('// DO NOT MODIFY MANUALLY')
  lines.push('')
  lines.push('package io.tether.qvac.sdk.generated.api')
  lines.push('')
  lines.push('import kotlinx.coroutines.flow.Flow')
  lines.push('import org.json.JSONObject')
  lines.push('')
  lines.push(...renderKotlinDataClasses(operations))
  lines.push('interface QvacGeneratedApiClient {')
  lines.push(...renderKotlinInterfaceMethods(operations))
  lines.push('}')
  lines.push('')
  lines.push('object QvacGeneratedApiContract {')
  lines.push('  val operations: List<String> = listOf(')
  lines.push(...renderKotlinOperationList(operations))
  lines.push('  )')
  lines.push('}')
  lines.push('')
  return `${lines.join('\n')}\n`
}

function sanitizeTomlKey(value: string): string {
  return value.replace(/[^a-zA-Z0-9]/g, '_')
}

function renderTomlVersions(dependencies: GeneratedDependency[]): string[] {
  return dependencies.map((dependency) => `${sanitizeTomlKey(dependency.packageName)} = "${dependency.version}"`)
}

function renderTomlLibraries(dependencies: GeneratedDependency[]): string[] {
  return dependencies.map((dependency) => {
    const key = sanitizeTomlKey(dependency.packageName)
    return `${key} = { module = "${dependency.packageName}", version.ref = "${key}" }`
  })
}

export function toGradleVersionCatalog(
  dependencies: GeneratedDependency[],
  manifest: GeneratedAndroidManifest
): string {
  const lines: string[] = []
  lines.push('# AUTO-GENERATED BY scripts/android/generate.ts')
  lines.push('# DO NOT MODIFY MANUALLY')
  lines.push('')
  lines.push('[versions]')
  lines.push(`qvac_sdk = "${manifest.sdk.version}"`)
  lines.push(...renderTomlVersions(dependencies))
  lines.push('')
  lines.push('[libraries]')
  lines.push(...renderTomlLibraries(dependencies))
  lines.push('')
  return `${lines.join('\n')}\n`
}
