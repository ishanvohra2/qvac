import test from 'node:test'
import assert from 'node:assert/strict'
import {
  toCamelCase,
  readMarkerRange,
  replaceMarkerBlock,
  renderOperationsBlock,
  renderKotlinClientBlock
} from '../scripts/generate-sample-bindings.mjs'

test('toCamelCase normalizes separators and lowercases the first segment', () => {
  assert.equal(toCamelCase('plugin_invoke-stream'), 'pluginInvokeStream')
  assert.equal(toCamelCase('loadModel'), 'loadModel')
  assert.equal(toCamelCase('LoadModel'), 'loadModel')
})

test('readMarkerRange returns the start/end offsets of the marker pair', () => {
  const content = 'a<start>inner<end>b'
  const { start, end } = readMarkerRange(content, '<start>', '<end>')
  assert.equal(content.slice(start, start + '<start>'.length), '<start>')
  assert.equal(content.slice(end), '<end>b')
})

test('readMarkerRange throws when markers are missing or out of order', () => {
  assert.throws(() => readMarkerRange('no markers', '<start>', '<end>'))
  assert.throws(() => readMarkerRange('<end>...<start>', '<start>', '<end>'))
})

test('replaceMarkerBlock replaces only the content between markers', () => {
  const content = 'head\n// <s>\nOLD\n// <e>\ntail'
  const replaced = replaceMarkerBlock(content, '// <s>', '// <e>', 'NEW')
  assert.equal(replaced, 'head\n// <s>\nNEW\n// <e>\ntail')
})

test('renderOperationsBlock emits a sorted-agnostic JS array literal', () => {
  const block = renderOperationsBlock(['loadModel', 'heartbeat'])
  assert.equal(block, "const CONTRACT_OPERATIONS = [\n  'loadModel',\n  'heartbeat',\n]")
})

test('renderKotlinClientBlock renders streaming and non-streaming overrides', () => {
  const block = renderKotlinClientBlock([
    {
      operation: 'loadModel',
      streaming: false,
      requestTypeName: 'LoadModelRequest',
      responseTypeName: 'LoadModelResponse'
    },
    {
      operation: 'completionStream',
      streaming: true,
      requestTypeName: 'CompletionStreamRequest',
      responseTypeName: 'CompletionStreamStreamEvent'
    }
  ])

  assert.ok(
    block.includes('override suspend fun loadModel(request: LoadModelRequest): LoadModelResponse =')
  )
  assert.ok(block.includes('LoadModelResponse(invokeContract("loadModel", request.payload))'))
  assert.ok(
    block.includes(
      'override fun completionStream(request: CompletionStreamRequest): Flow<CompletionStreamStreamEvent> ='
    )
  )
  assert.ok(block.includes('invokeContractStream("completionStream", request.payload).map { payload ->'))
})

test('renderKotlinClientBlock maps the reserved "suspend" operation to suspendOperation', () => {
  const block = renderKotlinClientBlock([
    {
      operation: 'suspend',
      streaming: false,
      requestTypeName: 'SuspendRequest',
      responseTypeName: 'SuspendResponse'
    }
  ])

  assert.ok(block.includes('override suspend fun suspendOperation(request: SuspendRequest)'))
  assert.ok(block.includes('invokeContract("suspend", request.payload)'))
})
