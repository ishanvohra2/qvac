import fs from 'fs/promises'
import path from 'path'
import { fileURLToPath, pathToFileURL } from 'url'

const scriptDir = fileURLToPath(new URL('.', import.meta.url))
const packageDir = path.resolve(scriptDir, '..')
const apiContractPath = path.join(packageDir, '../sdk/android/generated/api-contract.json')
const appJsPath = path.join(packageDir, 'sample-app/src/main/js/app.js')
const bridgePath = path.join(packageDir, 'sample-app/src/main/java/io/tether/qvac/sample/BareQvacBridge.kt')

export function toCamelCase(value) {
  return value
    .replace(/[^a-zA-Z0-9]+/g, ' ')
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .map((part, index) =>
      index === 0 ? part.charAt(0).toLowerCase() + part.slice(1) : part.charAt(0).toUpperCase() + part.slice(1)
    )
    .join('')
}

export function readMarkerRange(content, startMarker, endMarker) {
  const start = content.indexOf(startMarker)
  const end = content.indexOf(endMarker)
  if (start === -1 || end === -1 || end <= start) {
    throw new Error(`Missing or invalid markers: ${startMarker} .. ${endMarker}`)
  }
  return { start, end }
}

export function replaceMarkerBlock(content, startMarker, endMarker, generatedInner) {
  const { start, end } = readMarkerRange(content, startMarker, endMarker)
  const before = content.slice(0, start + startMarker.length)
  const after = content.slice(end)
  return `${before}\n${generatedInner}\n${after}`
}

export function renderOperationsBlock(operations) {
  const lines = ['const CONTRACT_OPERATIONS = [']
  for (const operation of operations) {
    lines.push(`  '${operation}',`)
  }
  lines.push(']')
  return lines.join('\n')
}

export function renderDispatchBlock(operations) {
  const byOperation = {
    heartbeat: [
      "if (msg.action === 'heartbeat') {",
      "  send({ id: msg.id, type: 'heartbeat', number: Date.now() })",
      '  return',
      '}'
    ],
    modelRegistryGetModel: [
      "if (msg.action === 'modelRegistryGetModel') {",
      '  handleModelRegistryGetModel(msg, payload)',
      '  return',
      '}'
    ],
    modelRegistryList: [
      "if (msg.action === 'modelRegistryList') {",
      '  handleModelRegistryList(msg)',
      '  return',
      '}'
    ],
    modelRegistrySearch: [
      "if (msg.action === 'modelRegistrySearch') {",
      '  handleModelRegistrySearch(msg, payload)',
      '  return',
      '}'
    ],
    pluginInvoke: [
      "if (msg.action === 'pluginInvoke') {",
      '  await handleContractPluginInvoke(msg, payload)',
      '  return',
      '}'
    ],
    pluginInvokeStream: [
      "if (msg.action === 'pluginInvokeStream') {",
      '  await handleContractPluginInvokeStream(msg, payload)',
      '  return',
      '}'
    ],
    resume: [
      "if (msg.action === 'resume') {",
      '  await resume()',
      "  send({ id: msg.id, type: 'resume' })",
      '  return',
      '}'
    ],
    state: [
      "if (msg.action === 'state') {",
      '  const currentState = await state()',
      "  send({ id: msg.id, type: 'state', state: currentState })",
      '  return',
      '}'
    ],
    suspend: [
      "if (msg.action === 'suspend') {",
      '  await suspend()',
      "  send({ id: msg.id, type: 'suspend' })",
      '  return',
      '}'
    ],
    unloadModel: [
      "if (msg.action === 'unloadModel') {",
      '  await handleUnloadModel(msg, payload)',
      '  return',
      '}'
    ],
    upscaleStream: [
      "if (msg.action === 'upscaleStream') {",
      '  await handleContractPluginInvokeStream(msg, {',
      '    modelId: payload.modelId,',
      "    handler: 'upscaleStream',",
      '    params: payload,',
      "    responseType: 'upscaleStream'",
      '  })',
      '  return',
      '}'
    ],
    vlaHparams: [
      "if (msg.action === 'vlaHparams') {",
      '  await handleContractPluginInvoke(msg, {',
      '    modelId: payload.modelId,',
      "    handler: 'vlaHparams',",
      '    params: payload',
      '  })',
      '  return',
      '}'
    ],
    vlaRun: [
      "if (msg.action === 'vlaRun') {",
      '  await handleContractPluginInvoke(msg, {',
      '    modelId: payload.modelId,',
      "    handler: 'vlaRun',",
      '    params: payload',
      '  })',
      '  return',
      '}'
    ]
  }

  const lines = []
  for (const operation of operations) {
    const snippet = byOperation[operation]
    if (!snippet) continue
    lines.push(...snippet)
  }
  return lines.map((line) => `    ${line}`).join('\n')
}

export function renderKotlinClientBlock(entries) {
  const lines = []
  for (const entry of entries) {
    const methodName = entry.operation === 'suspend' ? 'suspendOperation' : toCamelCase(entry.operation)
    if (entry.streaming) {
      lines.push(
        `    override fun ${methodName}(request: ${entry.requestTypeName}): Flow<${entry.responseTypeName}> =`,
        `      invokeContractStream("${entry.operation}", request.payload).map { payload ->`,
        `        ${entry.responseTypeName}(payload)`,
        '      }'
      )
    } else {
      lines.push(
        `    override suspend fun ${methodName}(request: ${entry.requestTypeName}): ${entry.responseTypeName} =`,
        `      ${entry.responseTypeName}(invokeContract("${entry.operation}", request.payload))`
      )
    }
    lines.push('')
  }
  if (lines.length > 0) {
    lines.pop()
  }
  return lines.join('\n')
}

async function writeIfChanged(filePath, nextContent, checkOnly) {
  let previous = null
  try {
    previous = await fs.readFile(filePath, 'utf8')
  } catch {
    previous = null
  }
  if (previous === nextContent) {
    console.log(`UNCHANGED ${path.relative(packageDir, filePath)}`)
    return false
  }
  if (checkOnly) {
    console.log(`OUTDATED ${path.relative(packageDir, filePath)}`)
    return true
  }
  await fs.writeFile(filePath, nextContent)
  console.log(`UPDATED ${path.relative(packageDir, filePath)}`)
  return true
}

async function main() {
  const checkOnly = process.argv.includes('--check')
  const contractRaw = await fs.readFile(apiContractPath, 'utf8')
  const contract = JSON.parse(contractRaw)
  const operations = contract.map((entry) => entry.operation).sort((a, b) => a.localeCompare(b))

  const appSource = await fs.readFile(appJsPath, 'utf8')
  const appWithOperations = replaceMarkerBlock(
    appSource,
    '// <generated-contract-operations:start>',
    '// <generated-contract-operations:end>',
    renderOperationsBlock(operations)
  )
  const appNext = replaceMarkerBlock(
    appWithOperations,
    '    // <generated-contract-dispatch:start>',
    '    // <generated-contract-dispatch:end>',
    renderDispatchBlock(operations)
  )

  const bridgeSource = await fs.readFile(bridgePath, 'utf8')
  const bridgeNext = replaceMarkerBlock(
    bridgeSource,
    '    // <generated-contract-client:start>',
    '    // <generated-contract-client:end>',
    renderKotlinClientBlock(contract)
  )

  let changed = 0
  if (await writeIfChanged(appJsPath, appNext, checkOnly)) changed += 1
  if (await writeIfChanged(bridgePath, bridgeNext, checkOnly)) changed += 1

  if (checkOnly && changed > 0) {
    process.exitCode = 1
    return
  }
  console.log(
    `android:generate-bindings ${checkOnly ? 'check' : 'complete'} (${changed} file${changed === 1 ? '' : 's'} changed)`
  )
}

const isMainModule = import.meta.url === pathToFileURL(process.argv[1] ?? '').href

if (isMainModule) {
  main().catch((error) => {
    console.error(error)
    process.exitCode = 1
  })
}
