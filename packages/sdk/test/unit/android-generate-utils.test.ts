import test from 'brittle'
import { androidManifestSourceSchema, type AndroidManifestSource } from '@/scripts/android/types'
import {
  collectApiOperationsFromSources,
  collectDependencies,
  shouldIncludeDependency,
  toCamelCase,
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

test('collectApiOperationsFromSources extracts operations via AST traversal', (t) => {
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
