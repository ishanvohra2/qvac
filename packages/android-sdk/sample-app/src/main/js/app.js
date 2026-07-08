import { completion, loadModel, unloadModel } from '@qvac/bare-sdk'
import { registerPlugin } from '@qvac/bare-sdk/plugins'
import { llmPlugin } from '@qvac/bare-sdk/llamacpp-completion/plugin'
import * as modelCatalog from '@qvac/bare-sdk/models'

registerPlugin(llmPlugin)

const { IPC } = BareKit
let loadedModelId = null
let activeStreamId = null

function send(message) {
  IPC.write(Buffer.from(`${JSON.stringify(message)}\n`))
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

async function handleLoadModel(msg) {
  sendLog('loadModel:start', { requestId: msg.id, modelSrc: msg.modelSrc ?? null })
  const modelSrc = resolveModelSource(msg.modelSrc)
  const modelId = await loadModel({
    modelSrc,
    modelType: 'llamacpp-completion',
    modelConfig: { ctx_size: 2048 }
  })
  loadedModelId = modelId
  sendLog('loadModel:success', { requestId: msg.id, modelId })
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
    if (msg.action === 'cancelStream') {
      handleCancel(msg)
      return
    }
    send({ id: msg.id, type: 'error', error: `Unsupported action: ${msg.action}` })
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    sendLog('action:error', { requestId: msg.id ?? null, action: msg.action ?? 'unknown', error: message })
    send({ id: msg.id ?? null, type: 'error', error: message })
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

sendLog('worklet:booted', { plugin: 'llamacpp-completion' })

Bare.on('uncaughtException', (error) => {
  sendLog('worklet:uncaughtException', {
    error: error instanceof Error ? error.message : String(error)
  })
  send({
    id: null,
    type: 'error',
    error: error instanceof Error ? error.message : String(error)
  })
})

Bare.on('unhandledRejection', (error) => {
  sendLog('worklet:unhandledRejection', {
    error: error instanceof Error ? error.message : String(error)
  })
})
