'use strict'

/**
 * qvac-side driver for the engine comparison harness. Runs under Bare
 * (not Node) because it loads the native transcription-parakeet addon.
 *
 * Loads a model ONCE, warms up, then runs N timed reps over each WAV clip,
 * capturing per-rep wall time (full JS run() path) and the transcript text.
 * Emits a JSON document the Node orchestrator (run-comparison.js) reads back.
 *
 * Invoked as:
 *   bare qvac-bench.js --model <gguf> --gpu true|false --threads N \
 *        --runs N --warmup N --out <json> --clips <wav1,wav2,...>
 */

const fs = require('bare-fs')
const process = require('bare-process')
const TranscriptionParakeet = require('../../index.js')

function parseArgs (argv) {
  const args = {}
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    if (a.startsWith('--')) { args[a.slice(2)] = argv[i + 1]; i++ }
  }
  return args
}

function nowMs () {
  const [s, ns] = process.hrtime()
  return s * 1000 + ns / 1e6
}

// Minimal RIFF/WAVE parser -> Float32Array of mono 16 kHz samples.
// Supports PCM s16le (downmixes multi-channel to mono).
function readWavFloat32 (path) {
  const buf = fs.readFileSync(path)
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength)
  if (dv.getUint32(0, false) !== 0x52494646) throw new Error(`not RIFF: ${path}`) // 'RIFF'
  if (dv.getUint32(8, false) !== 0x57415645) throw new Error(`not WAVE: ${path}`) // 'WAVE'
  let off = 12
  let channels = 1
  let bits = 16
  let dataOff = -1
  let dataLen = 0
  while (off + 8 <= buf.byteLength) {
    const id = dv.getUint32(off, false)
    const sz = dv.getUint32(off + 4, true)
    const body = off + 8
    if (id === 0x666d7420) { // 'fmt '
      channels = dv.getUint16(body + 2, true)
      bits = dv.getUint16(body + 14, true)
    } else if (id === 0x64617461) { // 'data'
      dataOff = body
      dataLen = sz
    }
    off = body + sz + (sz & 1)
  }
  if (dataOff < 0) throw new Error(`no data chunk: ${path}`)
  if (bits !== 16) throw new Error(`only PCM16 supported (got ${bits}-bit): ${path}`)
  const frames = (dataLen / 2 / channels) | 0
  const out = new Float32Array(frames)
  for (let f = 0; f < frames; f++) {
    let acc = 0
    for (let c = 0; c < channels; c++) {
      acc += dv.getInt16(dataOff + (f * channels + c) * 2, true)
    }
    out[f] = (acc / channels) / 32768.0
  }
  return out
}

async function transcribeOnce (model, audio) {
  const segs = []
  const response = await model.run(audio)
  await response.onUpdate(out => {
    const items = Array.isArray(out) ? out : [out]
    for (const s of items) if (s && s.text) segs.push(s.text)
  }).await()
  const stats = response.stats || null
  return { text: segs.join(' ').replace(/\s+/g, ' ').trim(), stats }
}

async function main () {
  const args = parseArgs(process.argv.slice(2))
  const useGPU = String(args.gpu) === 'true'
  const threads = Number.parseInt(args.threads || '4', 10)
  const runs = Number.parseInt(args.runs || '5', 10)
  const warmup = Number.parseInt(args.warmup || '1', 10)
  const clips = (args.clips || '').split(',').filter(Boolean)
  const SAMPLE_RATE = 16000

  const model = new TranscriptionParakeet({
    files: { model: args.model },
    config: { parakeetConfig: { maxThreads: threads, useGPU, sampleRate: SAMPLE_RATE, channels: 1 } }
  })

  const result = { model: args.model, useGPU, threads, backendId: null, clips: [] }
  try {
    await model.load()
    // Global warmup with silence to force full init.
    await transcribeOnce(model, new Float32Array(SAMPLE_RATE).fill(0)).catch(() => null)

    for (const clip of clips) {
      const audio = readWavFloat32(clip)
      const audioSec = audio.length / SAMPLE_RATE
      // Per-clip warmup (discarded).
      for (let w = 0; w < warmup; w++) await transcribeOnce(model, audio)
      const procMs = []
      let text = ''
      for (let i = 0; i < runs; i++) {
        const t0 = nowMs()
        const r = await transcribeOnce(model, audio)
        procMs.push(nowMs() - t0)
        text = r.text
        if (r.stats && typeof r.stats.backendId === 'number') result.backendId = r.stats.backendId
      }
      result.clips.push({ path: clip, audioSec, procMs, text })
    }
  } finally {
    try { await model.unload() } catch (_) {}
  }

  fs.writeFileSync(args.out, JSON.stringify(result, null, 2) + '\n')
}

main().then(() => process.exit(0)).catch(err => {
  console.error('qvac-bench error:', err && err.message ? err.message : err)
  process.exit(1)
})
