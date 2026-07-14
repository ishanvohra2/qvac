import test from 'brittle'
import {
  androidManifestSourceSchema,
  generatedAndroidManifestSchema,
  type AndroidManifestSource,
  type GeneratedApiOperation
} from '@/scripts/android/types'
import {
  buildCapabilities,
  buildModelConstants,
  collectApiOperationsFromSources,
  collectDependencies,
  mergeRegistryOperations,
  shouldIncludeDependency,
  toCamelCase,
  toGradleVersionCatalog,
  toKotlinApi,
  toKotlinFunctionName,
  toPascalCase
} from '@/scripts/android/generate-utils'

function makeManifestSource(): AndroidManifestSource {
  return androidManifestSourceSchema.parse({
    schemaVersion: 1,
    android: {
      groupId: 'io.tether.qvac',
      artifactId: 'android-sdk',
      namespace: 'io.tether.qvac.sdk',
      kotlinPackage: 'io.tether.qvac.sdk.generated',
      minSdk: 26,
      targetSdk: 35,
      compileSdk: 35,
      abis: ['arm64-v8a']
    },
    dependencyPolicy: {
      includeScopes: ['dependencies', 'peerDependencies'],
      includePrefixes: ['@qvac/', 'bare-'],
      excludePackages: ['@qvac/excluded']
    },
    runtime: {
      bareRuntimePackage: 'bare-runtime',
      bareAndroidTemplateRepo: 'holepunchto/bare-android-template'
    },
    addonPolicy: {
      llm: { androidSupported: true, fallbackBehavior: 'unsupported' },
      whisper: { androidSupported: true, fallbackBehavior: 'unsupported' },
      bci: { androidSupported: true, fallbackBehavior: 'unsupported' },
      embeddings: { androidSupported: true, fallbackBehavior: 'unsupported' },
      nmt: { androidSupported: true, fallbackBehavior: 'unsupported' },
      vad: { androidSupported: true, fallbackBehavior: 'unsupported' },
      tts: { androidSupported: true, fallbackBehavior: 'unsupported' },
      ocr: { androidSupported: true, fallbackBehavior: 'unsupported' },
      parakeet: { androidSupported: true, fallbackBehavior: 'unsupported' },
      diffusion: { androidSupported: true, fallbackBehavior: 'unsupported' },
      vla: { androidSupported: true, fallbackBehavior: 'unsupported' },
      classification: { androidSupported: true, fallbackBehavior: 'unsupported' },
      other: { androidSupported: false, fallbackBehavior: 'remote-only' }
    }
  })
}

test('shouldIncludeDependency respects include prefixes and excludes', (t) => {
  const source = makeManifestSource()

  t.is(shouldIncludeDependency('@qvac/sdk', source), true)
  t.is(shouldIncludeDependency('bare-runtime', source), true)
  t.is(shouldIncludeDependency('@qvac/excluded', source), false)
  t.is(shouldIncludeDependency('react', source), false)
})

test('collectDependencies filters and sorts dependencies from configured scopes', (t) => {
  const source = makeManifestSource()
  const dependencies = collectDependencies(
    {
      dependencies: {
        '@qvac/sdk': '^1.0.0',
        react: '^19.0.0',
        'bare-fs': '^4.0.0'
      },
      peerDependencies: {
        '@qvac/cli': '^1.0.0',
        '@qvac/excluded': '^9.9.9'
      }
    },
    source
  )

  t.alike(dependencies, [
    { packageName: '@qvac/cli', version: '^1.0.0', sourceScope: 'peerDependencies' },
    { packageName: '@qvac/sdk', version: '^1.0.0', sourceScope: 'dependencies' },
    { packageName: 'bare-fs', version: '^4.0.0', sourceScope: 'dependencies' }
  ])
})

test('case converters normalize operation names', (t) => {
  t.is(toPascalCase('pluginInvokeStream'), 'PluginInvokeStream')
  t.is(toPascalCase('plugin_invoke-stream'), 'PluginInvokeStream')
  t.is(toCamelCase('plugin_invoke-stream'), 'pluginInvokeStream')
})

test('collectApiOperationsFromSources extracts operations via regex schema parsing', (t) => {
  const operations = collectApiOperationsFromSources([
    {
      fileName: 'plugin.ts',
      content: `
import { z } from 'zod'

export const pluginInvokeRequestSchema = z.object({
  type: z.literal('pluginInvoke')
})
export const pluginInvokeResponseSchema = z.object({
  type:
    z
      .literal('pluginInvoke')
})

const sharedBase = z.object({})
export const pluginInvokeStreamRequestSchema = sharedBase.extend({
  type: z.literal('pluginInvokeStream')
})
export const pluginInvokeStreamResponseSchema = z.object({
  type: z.literal('pluginInvokeStream')
})
`
    },
    {
      fileName: 'sdcpp-config.ts',
      content: `
import { z } from 'zod'

export const upscaleStreamRequestSchema = z.object({
  type: z.literal('upscaleStream')
})
export const upscaleStreamResponseSchema = z.object({
  type: z.literal('upscaleStream')
})

export const ignoredRequestSchema = z.object({
  type: z.string()
})
`
    }
  ])

  t.alike(
    operations.map((entry) => ({
      operation: entry.operation,
      streaming: entry.streaming,
      requestTypeName: entry.requestTypeName,
      responseTypeName: entry.responseTypeName,
      sourceFile: entry.sourceFile
    })),
    [
      {
        operation: 'pluginInvoke',
        streaming: false,
        requestTypeName: 'PluginInvokeRequest',
        responseTypeName: 'PluginInvokeResponse',
        sourceFile: 'plugin.ts'
      },
      {
        operation: 'pluginInvokeStream',
        streaming: true,
        requestTypeName: 'PluginInvokeStreamRequest',
        responseTypeName: 'PluginInvokeStreamStreamEvent',
        sourceFile: 'plugin.ts'
      },
      {
        operation: 'upscaleStream',
        streaming: true,
        requestTypeName: 'UpscaleStreamRequest',
        responseTypeName: 'UpscaleStreamStreamEvent',
        sourceFile: 'sdcpp-config.ts'
      }
    ]
  )
})

test('collectApiOperationsFromSources ignores non-literal type discriminants without grabbing nested literals', (t) => {
  const operations = collectApiOperationsFromSources([
    {
      fileName: 'enum-type.ts',
      content: `
import { z } from 'zod'

export const quantizeRequestSchema = z.object({
  type: z.enum(['q4_k', 'q8_0']),
  nested: z.object({
    type: z.literal('shouldNotLeak')
  })
})

const transformBaseSchema = z.object({
  type: z.literal('shouldAlsoNotLeak')
})

export const optionsToRequestSchema = z
  .object({ value: z.string() })
  .transform((data) => ({
    type: 'shouldAlsoNotLeak' as const,
    value: data.value
  }))

export const describedRequestSchema = z.object({
  type: z.literal('described').describe('with a trailing chain'),
  value: z.string()
})
`
    }
  ])

  t.alike(
    operations.map((entry) => entry.operation),
    ['described']
  )
})

function makeGeneratedManifest() {
  const source = makeManifestSource()
  return generatedAndroidManifestSchema.parse({
    schemaVersion: 1,
    sourceSchemaVersion: 1,
    generatedAt: 'sdk-version-1.2.3',
    sdk: { packageName: '@qvac/sdk', version: '1.2.3' },
    android: source.android,
    runtime: source.runtime,
    dependencies: [],
    capabilities: []
  })
}

test('mergeRegistryOperations preserves schema entries and appends registry-only operations', (t) => {
  const fromSchemas: GeneratedApiOperation[] = [
    {
      operation: 'pluginInvoke',
      requestSchema: 'pluginInvokeRequestSchema',
      responseSchema: 'pluginInvokeResponseSchema',
      requestTypeName: 'PluginInvokeRequest',
      responseTypeName: 'PluginInvokeResponse',
      streaming: false,
      sourceFile: 'plugin.ts'
    }
  ]
  const registrySource = `
export const handlerRegistry = {
  pluginInvoke: { type: 'request' },
  loadModel: { type: 'request' },
  completionStream: { type: 'stream' },
  pluginInvokeStream: { type: 'duplex' }
}
`
  const merged = mergeRegistryOperations(fromSchemas, registrySource)

  t.alike(
    merged.map((entry) => entry.operation),
    ['completionStream', 'loadModel', 'pluginInvoke', 'pluginInvokeStream']
  )

  const pluginInvoke = merged.find((entry) => entry.operation === 'pluginInvoke')
  t.is(pluginInvoke?.sourceFile, 'plugin.ts')

  const loadModel = merged.find((entry) => entry.operation === 'loadModel')
  t.alike(loadModel, {
    operation: 'loadModel',
    requestSchema: 'loadModelRequestSchema',
    responseSchema: 'loadModelResponseSchema',
    requestTypeName: 'LoadModelRequest',
    responseTypeName: 'LoadModelResponse',
    streaming: false,
    sourceFile: 'handler-registry.ts'
  })

  const stream = merged.find((entry) => entry.operation === 'completionStream')
  t.is(stream?.streaming, true)
  t.is(stream?.responseSchema, 'completionStreamStreamResponseSchema')
  t.is(stream?.responseTypeName, 'CompletionStreamStreamEvent')

  const duplex = merged.find((entry) => entry.operation === 'pluginInvokeStream')
  t.is(duplex?.streaming, true)
})

test('toKotlinFunctionName suffixes reserved Kotlin words', (t) => {
  t.is(toKotlinFunctionName('loadModel'), 'loadModel')
  t.is(toKotlinFunctionName('suspend'), 'suspendOperation')
  t.is(toKotlinFunctionName('object'), 'objectOperation')
})

test('toKotlinApi renders wrappers, interface methods, and the contract list', (t) => {
  const operations: GeneratedApiOperation[] = [
    {
      operation: 'loadModel',
      requestSchema: 'loadModelRequestSchema',
      responseSchema: 'loadModelResponseSchema',
      requestTypeName: 'LoadModelRequest',
      responseTypeName: 'LoadModelResponse',
      streaming: false,
      sourceFile: 'load-model.ts'
    },
    {
      operation: 'completionStream',
      requestSchema: 'completionStreamRequestSchema',
      responseSchema: 'completionStreamStreamResponseSchema',
      requestTypeName: 'CompletionStreamRequest',
      responseTypeName: 'CompletionStreamStreamEvent',
      streaming: true,
      sourceFile: 'completion.ts'
    }
  ]

  const kotlin = toKotlinApi(operations)

  t.ok(kotlin.startsWith('// AUTO-GENERATED BY scripts/android/generate.ts\n'))
  t.ok(kotlin.includes('data class LoadModelRequest(val payload: JSONObject = JSONObject())'))
  t.ok(kotlin.includes('data class CompletionStreamStreamEvent(val payload: JSONObject = JSONObject())'))
  t.ok(kotlin.includes('  suspend fun loadModel(request: LoadModelRequest): LoadModelResponse'))
  t.ok(
    kotlin.includes(
      '  fun completionStream(request: CompletionStreamRequest): Flow<CompletionStreamStreamEvent>'
    )
  )
  t.ok(kotlin.includes('    "loadModel",'))
  t.absent(kotlin.includes('NOTE:'))
  t.ok(kotlin.endsWith('\n'))
})

test('toGradleVersionCatalog emits sanitized version refs and library entries', (t) => {
  const manifest = makeGeneratedManifest()
  const toml = toGradleVersionCatalog(
    [
      { packageName: '@qvac/sdk', version: '^1.0.0', sourceScope: 'dependencies' },
      { packageName: 'bare-fs', version: '^4.0.0', sourceScope: 'dependencies' }
    ],
    manifest
  )

  t.ok(toml.includes('qvac_sdk = "1.2.3"'))
  t.ok(toml.includes('_qvac_sdk = "^1.0.0"'))
  t.ok(toml.includes('_qvac_sdk = { module = "@qvac/sdk", version.ref = "_qvac_sdk" }'))
  t.ok(toml.includes('bare_fs = "^4.0.0"'))
  t.ok(toml.includes('bare_fs = { module = "bare-fs", version.ref = "bare_fs" }'))
})

test('buildCapabilities aggregates engines and model counts per configured addon', (t) => {
  const source = makeManifestSource()
  const capabilities = buildCapabilities(source, [
    { addon: 'llm', engine: 'llamacpp-completion' },
    { addon: 'llm', engine: 'llamacpp-embeddings' },
    { addon: 'whisper', engine: 'whispercpp-transcription' }
  ])

  const addons = capabilities.map((capability) => capability.addon)
  t.alike(addons, [...addons].sort((a, b) => a.localeCompare(b)))

  const llm = capabilities.find((capability) => capability.addon === 'llm')
  t.alike(llm?.engines, ['llamacpp-completion', 'llamacpp-embeddings'])
  t.is(llm?.modelCount, 2)

  const other = capabilities.find((capability) => capability.addon === 'other')
  t.is(other?.modelCount, 0)
  t.alike(other?.engines, [])
})

test('buildModelConstants builds registry src and sorts by name', (t) => {
  const constants = buildModelConstants([
    {
      name: 'ZED',
      registrySource: 's3',
      registryPath: 'path/z.bin',
      modelId: 'z.bin',
      addon: 'llm',
      engine: 'llamacpp-completion',
      quantization: 'Q4_K_M',
      params: '8B'
    },
    {
      name: 'ALPHA',
      registrySource: 's3',
      registryPath: 'path/a.bin',
      modelId: 'a.bin',
      addon: 'tts',
      engine: 'ggml-tts',
      quantization: '',
      params: ''
    }
  ])

  t.alike(
    constants.map((constant) => constant.name),
    ['ALPHA', 'ZED']
  )
  t.is(constants[0]?.src, 'registry://s3/path/a.bin')
})
