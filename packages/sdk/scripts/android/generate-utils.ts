import type {
  AndroidManifestSource,
  GeneratedApiOperation,
  GeneratedDependency
} from './types'
import { generatedApiOperationSchema } from './types'

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

export function collectDependencies(
  packageJson: PackageJson,
  source: AndroidManifestSource
): GeneratedDependency[] {
  const dependencies: GeneratedDependency[] = []

  for (const scopeName of source.dependencyPolicy.includeScopes) {
    const scoped = getScopeDependencies(packageJson, scopeName)
    for (const packageName of Object.keys(scoped)) {
      if (!shouldIncludeDependency(packageName, source)) continue
      dependencies.push({
        packageName,
        version: scoped[packageName]!,
        sourceScope: scopeName
      })
    }
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

function readOperationLiteral(schemaBlock: string): string | null {
  const typeLiteralRegex = /\btype\s*:\s*[\s\S]*?\bz\s*\.\s*literal\s*\(\s*(['"`])([^'"`]+)\1\s*\)/
  const match = typeLiteralRegex.exec(schemaBlock)
  if (!match) return null
  return match[2] ?? null
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
