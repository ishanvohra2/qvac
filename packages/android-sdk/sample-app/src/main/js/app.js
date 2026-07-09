import { completion, loadModel, unloadModel, textToSpeech, transcribe, translate } from '@qvac/bare-sdk'
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

async function handleLoadModel(msg) {
  const modelType = msg.modelType ?? 'llamacpp-completion'
  sendLog('loadModel:start', { requestId: msg.id, modelSrc: msg.modelSrc ?? null, modelType })
  const modelSrc = resolveModelSource(msg.modelSrc)
  const modelConfig = msg.modelConfig ?? resolveDefaultModelConfig(modelType)
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

async function handleUnloadModel(msg) {
  sendLog('unloadModel:start', { requestId: msg.id, loadedModelId })
  if (loadedModelId) {
    await unloadModel({ modelId: loadedModelId, autoClose: false })
  }
  loadedModelId = null
  sendLog('unloadModel:success', { requestId: msg.id })
  send({ id: msg.id, type: 'unloadModelResult', success: true })
}

async function handleCompletion(msg) {
  if (!loadedModelId) {
    send({ id: msg.id, type: 'error', error: 'No model loaded' })
    return
  }

  activeStreamId = msg.id
  sendLog('completion:start', { requestId: msg.id, modelId: loadedModelId })
  const run = completion({
    modelId: loadedModelId,
    history: [{ role: 'user', content: msg.prompt }],
    stream: true
  })

  for await (const token of run.tokenStream) {
    if (activeStreamId !== msg.id) break
    send({ id: msg.id, type: 'token', token })
  }

  if (activeStreamId === msg.id) {
    sendLog('completion:done', { requestId: msg.id })
    send({ id: msg.id, type: 'done' })
  }
}

async function handleTextToSpeech(msg) {
  if (!loadedModelId) {
    send({ id: msg.id, type: 'error', error: 'No model loaded' })
    return
  }
  const text = typeof msg.text === 'string' ? msg.text : ''
  const result = textToSpeech({
    modelId: loadedModelId,
    text,
    inputType: 'text',
    stream: false
  })
  const audioBuffer = await result.buffer
  const audioBytes = Buffer.from(audioBuffer.buffer, audioBuffer.byteOffset, audioBuffer.byteLength)
  const pcmBase64 = audioBytes.toString('base64')
  send({
    id: msg.id,
    type: 'textToSpeechResult',
    sampleCount: audioBuffer.length,
    sampleRate: 44100,
    pcmBase64,
    previewSamples: Array.from(audioBuffer.slice(0, 16))
  })
}

async function handleTranscribe(msg) {
  if (!loadedModelId) {
    send({ id: msg.id, type: 'error', error: 'No model loaded' })
    return
  }
  const text = await transcribe({
    modelId: loadedModelId,
    audioChunk: msg.audioChunk,
    ...(msg.prompt ? { prompt: msg.prompt } : {})
  })
  send({ id: msg.id, type: 'transcriptionResult', text })
}

async function handleTranslate(msg) {
  if (!loadedModelId) {
    send({ id: msg.id, type: 'error', error: 'No model loaded' })
    return
  }
  const result = translate({
    modelId: loadedModelId,
    text: msg.text,
    modelType: 'nmtcpp-translation',
    stream: false
  })
  const text = typeof result === 'string' ? result : await result.text
  send({ id: msg.id, type: 'translationResult', text })
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
      await handleUnloadModel(msg)
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
