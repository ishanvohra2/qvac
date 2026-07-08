import { z } from 'zod'

const modelRegistryEntryAddonSchema = z.enum([
  'llm',
  'whisper',
  'bci',
  'embeddings',
  'nmt',
  'vad',
  'tts',
  'ocr',
  'parakeet',
  'diffusion',
  'vla',
  'classification',
  'other'
])

const fallbackBehaviorSchema = z.enum(['unsupported', 'remote-only'])

const addonPolicyEntrySchema = z.object({
  androidSupported: z.boolean(),
  fallbackBehavior: fallbackBehaviorSchema
})

export const androidManifestSourceSchema = z.object({
  schemaVersion: z.literal(1),
  android: z.object({
    groupId: z.string().min(1),
    artifactId: z.string().min(1),
    namespace: z.string().min(1),
    kotlinPackage: z.string().min(1),
    minSdk: z.number().int().positive(),
    targetSdk: z.number().int().positive(),
    compileSdk: z.number().int().positive(),
    abis: z.array(z.string().min(1)).min(1)
  }),
  dependencyPolicy: z.object({
    includeScopes: z.array(z.enum(['dependencies', 'peerDependencies'])).min(1),
    includePrefixes: z.array(z.string().min(1)).min(1),
    excludePackages: z.array(z.string())
  }),
  runtime: z.object({
    bareRuntimePackage: z.string().min(1),
    bareAndroidTemplateRepo: z.string().min(1)
  }),
  addonPolicy: z.record(modelRegistryEntryAddonSchema, addonPolicyEntrySchema)
})

export const generatedDependencySchema = z.object({
  packageName: z.string(),
  version: z.string(),
  sourceScope: z.enum(['dependencies', 'peerDependencies'])
})

export const generatedAddonCapabilitySchema = z.object({
  addon: modelRegistryEntryAddonSchema,
  androidSupported: z.boolean(),
  fallbackBehavior: fallbackBehaviorSchema,
  engines: z.array(z.string()),
  modelCount: z.number().int().nonnegative()
})

export const generatedModelConstantSchema = z.object({
  name: z.string(),
  src: z.string(),
  modelId: z.string(),
  registryPath: z.string(),
  registrySource: z.string(),
  addon: modelRegistryEntryAddonSchema,
  engine: z.string(),
  quantization: z.string(),
  params: z.string()
})

export const generatedApiOperationSchema = z.object({
  operation: z.string(),
  requestSchema: z.string(),
  responseSchema: z.string().nullable(),
  requestTypeName: z.string(),
  responseTypeName: z.string(),
  streaming: z.boolean(),
  sourceFile: z.string()
})

export const generatedAndroidManifestSchema = z.object({
  schemaVersion: z.literal(1),
  sourceSchemaVersion: z.literal(1),
  generatedAt: z.string(),
  sdk: z.object({
    packageName: z.literal('@qvac/sdk'),
    version: z.string().min(1)
  }),
  android: androidManifestSourceSchema.shape.android,
  runtime: androidManifestSourceSchema.shape.runtime,
  dependencies: z.array(generatedDependencySchema),
  capabilities: z.array(generatedAddonCapabilitySchema)
})

export type AndroidManifestSource = z.infer<typeof androidManifestSourceSchema>
export type GeneratedDependency = z.infer<typeof generatedDependencySchema>
export type GeneratedAddonCapability = z.infer<typeof generatedAddonCapabilitySchema>
export type GeneratedModelConstant = z.infer<typeof generatedModelConstantSchema>
export type GeneratedApiOperation = z.infer<typeof generatedApiOperationSchema>
export type GeneratedAndroidManifest = z.infer<typeof generatedAndroidManifestSchema>
