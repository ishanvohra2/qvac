import {
  cancel,
  deleteCache,
  downloadAsset,
  getLoadedModelInfo,
  getModelInfo,
  invokePlugin,
  invokePluginStream,
  loadModel,
  loggingStream,
  ragChunk,
  ragCloseWorkspace,
  ragDeleteEmbeddings,
  ragDeleteWorkspace,
  ragIngest,
  ragListWorkspaces,
  ragReindex,
  ragSaveEmbeddings,
  ragSearch,
  startQVACProvider,
  state,
  stopQVACProvider,
  suspend,
  resume,
  unloadModel
} from '@qvac/bare-sdk'
import { registerPlugin } from '@qvac/bare-sdk/plugins'
import { llmPlugin } from '@qvac/bare-sdk/llamacpp-completion/plugin'
import { ttsPlugin } from '@qvac/bare-sdk/tts-ggml/plugin'
import { whisperPlugin } from '@qvac/bare-sdk/whispercpp-transcription/plugin'
import { nmtPlugin } from '@qvac/bare-sdk/nmtcpp-translation/plugin'
import * as modelCatalog from '@qvac/bare-sdk/models'

registerPlugin(llmPlugin)
registerPlugin(ttsPlugin)
registerPlugin(whisperPlugin)
registerPlugin(nmtPlugin)

const { IPC } = BareKit
let loadedModelId = null
let activeStreamId = null
let activeTtsSampleRate = null

// <generated-contract-operations:start>
const CONTRACT_OPERATIONS = [
  'batchCompletionStream',
  'bciTranscribe',
  'bciTranscribeStream',
  'cancel',
  'classify',
  'completionStream',
  'deleteCache',
  'diffusionStream',
  'downloadAsset',
  'embed',
  'finetune',
  'getLoadedModelInfo',
  'getModelInfo',
  'heartbeat',
  'loadModel',
  'loggingStream',
  'modelRegistryGetModel',
  'modelRegistryList',
  'modelRegistrySearch',
  'ocrStream',
  'pluginInvoke',
  'pluginInvokeStream',
  'provide',
  'rag',
  'resume',
  'state',
  'stopProvide',
  'suspend',
  'textToSpeech',
  'textToSpeechStream',
  'transcribe',
  'transcribeStream',
  'translate',
  'unloadModel',
  'upscaleStream',
  'videoStream',
  'vlaHparams',
  'vlaRun',
]
// <generated-contract-operations:end>

function getCatalogEntries() {
  return Object.entries(modelCatalog)
    .filter(([name, value]) => {
      return (
        typeof name === 'string' &&
        value &&
        typeof value === 'object' &&
        typeof value.src === 'string' &&
        typeof value.modelId === 'string'
      )
    })
    .map(([name, value]) => ({
      name,
      ...value
    }))
}

function resolveRequestModelId(payload) {
  const explicitModelId = typeof payload.modelId === 'string' ? payload.modelId : null
  if (explicitModelId) return explicitModelId
  return loadedModelId
}

function send(message) {
  IPC.write(Buffer.from(`${JSON.stringify(message)}\n`))
}

function normalizeErrorDetails(error) {
  if (!(error instanceof Error)) {
    return {
      message: String(error),
      name: 'NonErrorThrow',
      code: null,
      stack: null,
      cause: null
    }
  }
  const typed = error
  return {
    message: error.message || 'Unknown error',
    name: error.name || 'Error',
    code: typeof typed.code === 'string' ? typed.code : null,
    stack: typeof error.stack === 'string' ? error.stack : null,
    cause: error.cause ? String(error.cause) : null
  }
}

function sendLog(message, details = {}) {
  send({
    id: null,
    type: 'log',
    message,
    ...details
  })
}

function resolveModelSource(input) {
  if (!input) return modelCatalog.LLAMA_3_2_1B_INST_Q4_0
  const constant = modelCatalog[input]
  if (constant && typeof constant === 'object' && typeof constant.src === 'string') {
    return constant
  }
  return input
}

function resolveDefaultModelConfig(modelType) {
  if (modelType === 'llamacpp-completion') {
    return { ctx_size: 2048 }
  }
  if (modelType === 'whispercpp-transcription') {
    return { audio_format: 'f32le' }
  }
  return {}
}

function resolveTtsSampleRate(config) {
  if (!config || typeof config !== 'object') return null
  const outputSampleRate = config.outputSampleRate
  if (typeof outputSampleRate === 'number' && Number.isFinite(outputSampleRate) && outputSampleRate > 0) {
    return Math.round(outputSampleRate)
  }
  return null
}

async function handleLoadModel(msg) {
  const modelType = msg.modelType ?? 'llamacpp-completion'
  sendLog('loadModel:start', { requestId: msg.id, modelSrc: msg.modelSrc ?? null, modelType })
  const modelSrc = resolveModelSource(msg.modelSrc)
  const modelConfig = msg.modelConfig ?? resolveDefaultModelConfig(modelType)
  if (modelType === 'tts-ggml') {
    activeTtsSampleRate = resolveTtsSampleRate(modelConfig)
  } else {
    activeTtsSampleRate = null
  }
  const modelId = await loadModel({
    modelSrc,
    modelType,
    modelConfig,
    onProgress: (progress) => {
      sendLog('loadModel:progress', {
        requestId: msg.id,
        progressType: progress?.type ?? 'unknown',
        downloadedBytes: progress?.downloadedBytes ?? null,
        totalBytes: progress?.totalBytes ?? null
      })
    }
  })
  loadedModelId = modelId
  sendLog('loadModel:success', { requestId: msg.id, modelId, modelType })
  send({ id: msg.id, type: 'loadModelResult', success: true, modelId })
}

async function handleUnloadModel(msg, payload = {}) {
  sendLog('unloadModel:start', { requestId: msg.id, loadedModelId })
  const modelId = resolveRequestModelId(payload)
  if (modelId) {
    await unloadModel({ modelId, autoClose: false })
  }
  loadedModelId = null
  activeTtsSampleRate = null
  sendLog('unloadModel:success', { requestId: msg.id, modelId })
  send({ id: msg.id, type: 'unloadModel', success: true, hasActiveModels: false })
}

async function handleCompletion(msg) {
  const modelId = resolveRequestModelId(msg)
  if (!modelId) {
    send({ id: msg.id, type: 'error', error: 'No model loaded' })
    return
  }

  activeStreamId = msg.id
  sendLog('completion:start', { requestId: msg.id, modelId })
  const run = invokePluginStream({
    modelId,
    handler: 'completionStream',
    params: {
      type: 'completionStream',
      modelId,
      history: [{ role: 'user', content: msg.prompt }],
      stream: true
    }
  })

  for await (const chunk of run) {
    if (activeStreamId !== msg.id) break
    const events = Array.isArray(chunk?.events) ? chunk.events : []
    for (const event of events) {
      if (event?.type === 'contentDelta' && typeof event.text === 'string') {
        send({ id: msg.id, type: 'token', token: event.text })
      }
    }
  }

  if (activeStreamId === msg.id) {
    sendLog('completion:done', { requestId: msg.id })
    send({ id: msg.id, type: 'done' })
  }
}

async function handleTextToSpeech(msg) {
  const modelId = resolveRequestModelId(msg)
  if (!modelId) {
    send({ id: msg.id, type: 'error', error: 'No model loaded' })
    return
  }
  const text = typeof msg.text === 'string' ? msg.text : ''
  const stream = invokePluginStream({
    modelId,
    handler: 'textToSpeech',
    params: {
      type: 'textToSpeech',
      modelId,
      text,
      inputType: 'text',
      stream: true
    }
  })
  const merged = []
  for await (const chunk of stream) {
    if (Array.isArray(chunk?.buffer) && chunk.buffer.length > 0) {
      merged.push(...chunk.buffer)
    }
  }
  const sampleArray = Int16Array.from(merged)
  const audioBytes = Buffer.from(
    sampleArray.buffer,
    sampleArray.byteOffset,
    sampleArray.byteLength
  )
  const pcmBase64 = audioBytes.toString('base64')
  const requestedSampleRate =
    typeof msg.sampleRate === 'number' && Number.isFinite(msg.sampleRate) && msg.sampleRate > 0
      ? Math.round(msg.sampleRate)
      : null
  const sampleRate = requestedSampleRate ?? activeTtsSampleRate
  send({
    id: msg.id,
    type: 'textToSpeechResult',
    sampleCount: merged.length,
    ...(sampleRate !== null ? { sampleRate } : {}),
    pcmBase64,
    previewSamples: merged.slice(0, 16)
  })
}

async function handleTranscribe(msg) {
  const modelId = resolveRequestModelId(msg)
  if (!modelId) {
    send({ id: msg.id, type: 'error', error: 'No model loaded' })
    return
  }
  const result = await invokePlugin({
    modelId,
    handler: 'transcribe',
    params: {
      type: 'transcribe',
      modelId,
      audioChunk: {
        type: 'filePath',
        value: msg.audioChunk
      },
      ...(msg.prompt ? { prompt: msg.prompt } : {})
    }
  })
  const text = typeof result?.text === 'string' ? result.text : ''
  send({ id: msg.id, type: 'transcriptionResult', text })
}

async function handleTranslate(msg) {
  const modelId = resolveRequestModelId(msg)
  if (!modelId) {
    send({ id: msg.id, type: 'error', error: 'No model loaded' })
    return
  }
  const stream = invokePluginStream({
    modelId,
    handler: 'translate',
    params: {
      type: 'translate',
      modelId,
      text: msg.text,
      modelType: 'nmtcpp-translation',
      stream: true
    }
  })
  let text = ''
  for await (const chunk of stream) {
    if (typeof chunk?.token === 'string') {
      text += chunk.token
    }
  }
  send({ id: msg.id, type: 'translationResult', text })
}

async function handleContractPluginInvoke(msg, payload) {
  const modelId = resolveRequestModelId(payload)
  if (!modelId) {
    send({ id: msg.id, type: 'error', error: 'No model loaded' })
    return
  }
  const result = await invokePlugin({
    modelId,
    handler: payload.handler,
    params: payload.params
  })
  send({
    id: msg.id,
    type: 'pluginInvoke',
    result
  })
}

async function handleContractPluginInvokeStream(msg, payload) {
  const modelId = resolveRequestModelId(payload)
  if (!modelId) {
    send({ id: msg.id, type: 'error', error: 'No model loaded' })
    return
  }
  for await (const chunk of invokePluginStream({
    modelId,
    handler: payload.handler,
    params: payload.params
  })) {
    send({
      id: msg.id,
      type: payload.responseType ?? 'pluginInvokeStream',
      result: chunk,
      done: false
    })
  }
  send({
    id: msg.id,
    type: payload.responseType ?? 'pluginInvokeStream',
    result: null,
    done: true
  })
}

function handleModelRegistryList(msg) {
  send({
    id: msg.id,
    type: 'modelRegistryList',
    models: getCatalogEntries()
  })
}

function handleModelRegistryGetModel(msg, payload) {
  const modelName = typeof payload?.name === 'string' ? payload.name : ''
  const model =
    getCatalogEntries().find((entry) => entry.name === modelName) ??
    getCatalogEntries().find((entry) => entry.modelId === modelName) ??
    null
  send({
    id: msg.id,
    type: 'modelRegistryGetModel',
    model
  })
}

function handleModelRegistrySearch(msg, payload) {
  const query = String(payload?.query ?? '').toLowerCase()
  const models = getCatalogEntries().filter((entry) => {
    if (query.length === 0) return true
    return (
      entry.name.toLowerCase().includes(query) ||
      String(entry.modelId).toLowerCase().includes(query) ||
      String(entry.engine).toLowerCase().includes(query) ||
      String(entry.addon).toLowerCase().includes(query)
    )
  })
  send({
    id: msg.id,
    type: 'modelRegistrySearch',
    models
  })
}

function withOperationType(payload, operation) {
  const next = payload && typeof payload === 'object' ? { ...payload } : {}
  if (!next.type) {
    next.type = operation
  }
  return next
}

function isStreamOperation(operation) {
  return operation.endsWith('Stream')
}

async function handleRagOperation(payload) {
  const operation = payload.operation
  if (operation === 'chunk') return ragChunk(payload)
  if (operation === 'ingest') return ragIngest(payload)
  if (operation === 'saveEmbeddings') return ragSaveEmbeddings(payload)
  if (operation === 'search') return ragSearch(payload)
  if (operation === 'deleteEmbeddings') {
    await ragDeleteEmbeddings(payload)
    return { success: true }
  }
  if (operation === 'reindex') return ragReindex(payload)
  if (operation === 'listWorkspaces') return ragListWorkspaces()
  if (operation === 'closeWorkspace') {
    await ragCloseWorkspace(payload)
    return { success: true }
  }
  if (operation === 'deleteWorkspace') {
    await ragDeleteWorkspace(payload)
    return { success: true }
  }
  throw new Error(`Unsupported rag operation: ${String(operation)}`)
}

async function handleSdkContractOperation(msg, payload) {
  const operation = msg.action
  if (operation === 'cancel') {
    await cancel(payload)
    send({ id: msg.id, type: operation, success: true })
    return
  }
  if (operation === 'deleteCache') {
    const result = await deleteCache(payload)
    send({ id: msg.id, type: operation, ...result })
    return
  }
  if (operation === 'downloadAsset') {
    const assetId = await downloadAsset(payload)
    send({ id: msg.id, type: operation, success: true, assetId })
    return
  }
  if (operation === 'getModelInfo') {
    const result = await getModelInfo(payload)
    send({ id: msg.id, type: operation, payload: result })
    return
  }
  if (operation === 'getLoadedModelInfo') {
    const result = await getLoadedModelInfo(payload)
    send({ id: msg.id, type: operation, payload: result })
    return
  }
  if (operation === 'provide') {
    const result = await startQVACProvider(payload)
    send({ id: msg.id, type: operation, ...result })
    return
  }
  if (operation === 'stopProvide') {
    const result = await stopQVACProvider()
    send({ id: msg.id, type: operation, ...result })
    return
  }
  if (operation === 'loggingStream') {
    for await (const event of loggingStream(payload)) {
      send({ id: msg.id, type: operation, payload: event, done: false })
    }
    send({ id: msg.id, type: operation, payload: null, done: true })
    return
  }
  if (operation === 'rag') {
    const result = await handleRagOperation(payload)
    if (result && typeof result === 'object' && !Array.isArray(result)) {
      send({ id: msg.id, type: operation, ...result })
    } else {
      send({ id: msg.id, type: operation, payload: { result } })
    }
    return
  }

  const params = withOperationType(payload, operation)
  const modelId = resolveRequestModelId(params)
  if (!modelId) {
    throw new Error(`Operation ${operation} requires modelId`)
  }
  if (!params.modelId) {
    params.modelId = modelId
  }

  if (isStreamOperation(operation)) {
    await handleContractPluginInvokeStream(msg, {
      modelId,
      handler: operation,
      params,
      responseType: operation
    })
    return
  }

  const result = await invokePlugin({
    modelId,
    handler: operation,
    params
  })
  send({ id: msg.id, type: operation, payload: { result } })
}

function handleCancel(msg) {
  if (activeStreamId === msg.id) {
    activeStreamId = null
  }
  sendLog('completion:cancelled', { requestId: msg.id })
  send({ id: msg.id, type: 'cancelled' })
}

async function dispatch(msg) {
  try {
    const payload = msg.payload && typeof msg.payload === 'object' ? msg.payload : {}
    if (msg.action === 'health') {
      send({
        id: msg.id,
        type: 'healthResult',
        success: true,
        runtime: 'bare-worklet',
        plugin: 'llamacpp-completion',
        loadedModelId
      })
      return
    }
    if (msg.action === 'loadModel') {
      await handleLoadModel(msg)
      return
    }
    if (msg.action === 'unloadModel') {
      await handleUnloadModel(msg, payload)
      return
    }
    if (msg.action === 'completionStream') {
      await handleCompletion(msg)
      return
    }
    if (msg.action === 'textToSpeech') {
      await handleTextToSpeech(msg)
      return
    }
    if (msg.action === 'transcribe') {
      await handleTranscribe(msg)
      return
    }
    if (msg.action === 'translate') {
      await handleTranslate(msg)
      return
    }
    if (msg.action === 'cancelStream') {
      handleCancel(msg)
      return
    }
    // <generated-contract-dispatch:start>
    if (msg.action === 'heartbeat') {
      send({ id: msg.id, type: 'heartbeat', number: Date.now() })
      return
    }
    if (msg.action === 'modelRegistryGetModel') {
      handleModelRegistryGetModel(msg, payload)
      return
    }
    if (msg.action === 'modelRegistryList') {
      handleModelRegistryList(msg)
      return
    }
    if (msg.action === 'modelRegistrySearch') {
      handleModelRegistrySearch(msg, payload)
      return
    }
    if (msg.action === 'pluginInvoke') {
      await handleContractPluginInvoke(msg, payload)
      return
    }
    if (msg.action === 'pluginInvokeStream') {
      await handleContractPluginInvokeStream(msg, payload)
      return
    }
    if (msg.action === 'resume') {
      await resume()
      send({ id: msg.id, type: 'resume' })
      return
    }
    if (msg.action === 'state') {
      const currentState = await state()
      send({ id: msg.id, type: 'state', state: currentState })
      return
    }
    if (msg.action === 'suspend') {
      await suspend()
      send({ id: msg.id, type: 'suspend' })
      return
    }
    if (msg.action === 'unloadModel') {
      await handleUnloadModel(msg, payload)
      return
    }
    if (msg.action === 'upscaleStream') {
      await handleContractPluginInvokeStream(msg, {
        modelId: payload.modelId,
        handler: 'upscaleStream',
        params: payload,
        responseType: 'upscaleStream'
      })
      return
    }
    if (msg.action === 'vlaHparams') {
      await handleContractPluginInvoke(msg, {
        modelId: payload.modelId,
        handler: 'vlaHparams',
        params: payload
      })
      return
    }
    if (msg.action === 'vlaRun') {
      await handleContractPluginInvoke(msg, {
        modelId: payload.modelId,
        handler: 'vlaRun',
        params: payload
      })
      return
    }
    // <generated-contract-dispatch:end>
    if (CONTRACT_OPERATIONS.includes(msg.action)) {
      await handleSdkContractOperation(msg, payload)
      return
    }
    send({ id: msg.id, type: 'error', error: `Unsupported action: ${msg.action}` })
  } catch (error) {
    const details = normalizeErrorDetails(error)
    sendLog('action:error', {
      requestId: msg.id ?? null,
      action: msg.action ?? 'unknown',
      error: details.message,
      errorCode: details.code,
      errorName: details.name
    })
    send({
      id: msg.id ?? null,
      type: 'error',
      error: details.message,
      errorName: details.name,
      errorCode: details.code,
      errorStack: details.stack,
      errorCause: details.cause
    })
  }
}

IPC.on('data', async (data) => {
  try {
    const msg = JSON.parse(Buffer.from(data).toString('utf8'))
    await dispatch(msg)
  } catch (error) {
    send({
      id: null,
      type: 'error',
      error: error instanceof Error ? error.message : String(error)
    })
  }
})

sendLog('worklet:booted', {
  plugins: [
    'llamacpp-completion',
    'tts-ggml',
    'whispercpp-transcription',
    'nmtcpp-translation'
  ]
})

Bare.on('uncaughtException', (error) => {
  const details = normalizeErrorDetails(error)
  sendLog('worklet:uncaughtException', {
    error: details.message,
    errorCode: details.code,
    errorName: details.name
  })
  send({
    id: null,
    type: 'error',
    error: details.message,
    errorName: details.name,
    errorCode: details.code,
    errorStack: details.stack,
    errorCause: details.cause
  })
})

Bare.on('unhandledRejection', (error) => {
  const details = normalizeErrorDetails(error)
  sendLog('worklet:unhandledRejection', {
    error: details.message,
    errorCode: details.code,
    errorName: details.name
  })
})
