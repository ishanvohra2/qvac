#!/usr/bin/env node
'use strict'

/**
 * Parakeet engine comparison harness: qvac transcription-parakeet (ggml addon)
 * vs mudler/parakeet.cpp (parakeet-cli).
 *
 * Both engines run the SAME checkpoints at the SAME quant (q8_0) on the SAME
 * canonical 16 kHz mono clips, each from its own native GGUF (the two GGUF
 * schemas are not interchangeable -- different tensor names + KV metadata).
 *
 * Measures, per model x backend x clip:
 *   - processing time / RTF (load-once + warmup + N timed reps)
 *   - transcript text -> WER (vs reference where available; cross-engine
 *     agreement otherwise)
 *
 * Fairness notes (surfaced in the report):
 *   - Both timings are engine-only C++ inference (mel + encoder + decoder),
 *     excluding model load + wav read: qvac = parakeet-cpp `--bench`
 *     inference_ms; mudler = parakeet-cli `bench` transcribe_pcm. No JS/Bare
 *     tax on either side.
 *   - Same clips, same threads, same quant LEVEL (q8_0). The two q8_0
 *     converters keep slightly different tensor sets in F32 (file sizes differ).
 *
 * Usage:
 *   node run-comparison.js
 *   QVAC_CMP_RUNS=10 QVAC_CMP_MODELS=tdt,ctc QVAC_CMP_GPU=false node run-comparison.js
 */

const fs = require('fs')
const path = require('path')
const { spawnSync } = require('child_process')

const PKG_DIR = path.resolve(__dirname, '../..')
const MUDLER_DIR = path.resolve(PKG_DIR, '../parakeet.cpp')
const MUDLER_CLI = path.join(MUDLER_DIR, 'build/examples/cli/parakeet-cli')
const MUDLER_MODELS = path.join(MUDLER_DIR, 'models-gguf')
const QVAC_MODELS = path.join(PKG_DIR, 'models')

// qvac's own parakeet-cpp C++ engine CLI (binary name `parakeet`), built from
// tetherto/qvac-ext-lib-whisper.cpp/parakeet-cpp. We benchmark this directly
// (engine-only `--bench` inference_ms) instead of the Bare/Node addon, so the
// timing is apples-to-apples with mudler's engine-only `bench`.
const QVAC_ENGINE_DIR = process.env.QVAC_ENGINE_DIR || path.resolve(PKG_DIR, '../qvac-ext-lib-whisper.cpp/parakeet-cpp')
const QVAC_CLI = process.env.QVAC_PARAKEET_CLI ||
  ['build/parakeet', 'build-metal/parakeet', 'build-vk/parakeet', 'build-cl/parakeet']
    .map(p => path.join(QVAC_ENGINE_DIR, p)).find(p => fs.existsSync(p)) ||
  path.join(QVAC_ENGINE_DIR, 'build/parakeet')
const SAMPLES = path.join(PKG_DIR, 'examples/samples')
const OUT_DIR = path.join(__dirname, 'out')
const CLIP_DIR = path.join(OUT_DIR, 'clips')
const SAMPLE_RATE = 16000

const RUNS = intEnv('QVAC_CMP_RUNS', 5)
const WARMUP = intEnv('QVAC_CMP_WARMUP', 1)
const THREADS = intEnv('QVAC_CMP_THREADS', 4)
const QUANT = process.env.QVAC_CMP_QUANT || 'q8_0'

// Platform-derived presentation values. These are re-derivable from a stored
// meta.platform so a report can be re-rendered for any platform (e.g. on a Mac
// from a committed linux-x64 data file) -- see QVAC_CMP_RENDER_FROM in main().
let PLATFORM_SLUG, GPU_LABEL, PLATFORM_NOTE
function derivePlatform (platformStr) {
  const parts = (platformStr || `${process.platform}-${process.arch}`).split('-')
  const os = parts[0]
  const arch = parts.slice(1).join('-') || process.arch
  const osSlug = os === 'darwin' ? 'mac' : os === 'win32' ? 'windows' : os
  PLATFORM_SLUG = `${osSlug}-${arch}`
  GPU_LABEL = process.env.QVAC_CMP_GPU_LABEL || (os === 'darwin' ? 'Metal' : 'GPU')
  PLATFORM_NOTE = process.env.QVAC_CMP_PLATFORM_NOTE || (os === 'darwin' ? 'Apple Silicon, Metal' : `${osSlug}-${arch}, ${GPU_LABEL}`)
}
derivePlatform(`${process.platform}-${process.arch}`)

const MODEL_MAP = {
  tdt: { qvac: 'parakeet-tdt-0.6b-v3.q8_0.gguf', mudler: 'tdt-0.6b-v3-q8_0.gguf', decoder: 'tdt', multilingual: true, label: 'TDT 0.6B v3 (multilingual)' },
  ctc: { qvac: 'parakeet-ctc-0.6b.q8_0.gguf', mudler: 'ctc-0.6b-q8_0.gguf', decoder: 'ctc', multilingual: false, label: 'CTC 0.6B (English)' },
  eou: { qvac: 'parakeet-eou-120m-v1.q8_0.gguf', mudler: 'realtime_eou_120m-v1-q8_0.gguf', decoder: null, multilingual: false, label: 'EOU 120M v1 (rnnt/streaming, English)' }
}

const ALICE_REF = 'Alice was beginning to get very tired of sitting by her sister on the bank and of having nothing to do. Once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it. And what is the use of a book thought Alice without pictures or conversations'
const JFK_REF = 'And so my fellow Americans ask not what your country can do for you ask what you can do for your country'

// source: file in examples/samples; raw files are s16le 16 kHz mono.
const CLIPS = [
  { id: 'jfk', lang: 'en', src: 'jfk.wav', ref: JFK_REF },
  { id: 'alice', lang: 'en', src: 'sample-16k.wav', ref: ALICE_REF },
  { id: 'croatian', lang: 'hr', src: 'croatian.raw', ref: null },
  { id: 'french', lang: 'fr', src: 'French.raw', ref: null },
  { id: 'spanish60', lang: 'es', src: 'LastQuestion_long_ES.raw', ref: null, maxSec: 60 }
]

const MODELS = (process.env.QVAC_CMP_MODELS || 'tdt,ctc,eou').split(',').map(s => s.trim().toLowerCase()).filter(Boolean)
const GPU_MODES = process.env.QVAC_CMP_GPU === 'false' ? [false]
  : (process.env.QVAC_CMP_GPU === 'true' ? [true] : [false, true])

function intEnv (name, dflt) {
  const v = process.env[name]
  if (v === undefined) return dflt
  const n = Number.parseInt(v, 10)
  return Number.isNaN(n) ? dflt : n
}

// ---------------------------------------------------------------------------
// audio: normalise every source to canonical 16 kHz mono PCM16 WAV so both
// engines read byte-identical inputs.
// ---------------------------------------------------------------------------
function readSourceInt16 (srcPath) {
  if (srcPath.toLowerCase().endsWith('.wav')) {
    const buf = fs.readFileSync(srcPath)
    const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength)
    let off = 12; let channels = 1; let bits = 16; let dataOff = -1; let dataLen = 0
    while (off + 8 <= buf.byteLength) {
      const id = dv.getUint32(off, false); const sz = dv.getUint32(off + 4, true); const body = off + 8
      if (id === 0x666d7420) { channels = dv.getUint16(body + 2, true); bits = dv.getUint16(body + 14, true) } else if (id === 0x64617461) { dataOff = body; dataLen = sz }
      off = body + sz + (sz & 1)
    }
    if (dataOff < 0 || bits !== 16) throw new Error(`unsupported wav (need PCM16): ${srcPath}`)
    const frames = (dataLen / 2 / channels) | 0
    const out = new Int16Array(frames)
    for (let f = 0; f < frames; f++) {
      let acc = 0
      for (let c = 0; c < channels; c++) acc += dv.getInt16(dataOff + (f * channels + c) * 2, true)
      out[f] = Math.max(-32768, Math.min(32767, Math.round(acc / channels)))
    }
    return out
  }
  // raw s16le mono
  const buf = fs.readFileSync(srcPath)
  return new Int16Array(buf.buffer, buf.byteOffset, (buf.byteLength / 2) | 0)
}

function writeCanonicalWav (int16, outPath) {
  const dataLen = int16.length * 2
  const header = Buffer.alloc(44)
  header.write('RIFF', 0, 'ascii'); header.writeUInt32LE(36 + dataLen, 4); header.write('WAVE', 8, 'ascii')
  header.write('fmt ', 12, 'ascii'); header.writeUInt32LE(16, 16); header.writeUInt16LE(1, 20)
  header.writeUInt16LE(1, 22); header.writeUInt32LE(SAMPLE_RATE, 24); header.writeUInt32LE(SAMPLE_RATE * 2, 28)
  header.writeUInt16LE(2, 32); header.writeUInt16LE(16, 34)
  header.write('data', 36, 'ascii'); header.writeUInt32LE(dataLen, 40)
  const body = Buffer.from(int16.buffer, int16.byteOffset, dataLen)
  fs.writeFileSync(outPath, Buffer.concat([header, body]))
}

function buildClips () {
  if (!fs.existsSync(CLIP_DIR)) fs.mkdirSync(CLIP_DIR, { recursive: true })
  for (const clip of CLIPS) {
    let int16 = readSourceInt16(path.join(SAMPLES, clip.src))
    if (clip.maxSec) {
      const maxSamples = clip.maxSec * SAMPLE_RATE
      if (int16.length > maxSamples) int16 = int16.subarray(0, maxSamples)
    }
    clip.wav = path.join(CLIP_DIR, `${clip.id}.wav`)
    clip.audioSec = int16.length / SAMPLE_RATE
    writeCanonicalWav(int16, clip.wav)
  }
}

// ---------------------------------------------------------------------------
// stats + WER
// ---------------------------------------------------------------------------
function percentile (sorted, p) {
  if (!sorted.length) return 0
  const idx = (p / 100) * (sorted.length - 1); const lo = Math.floor(idx); const hi = Math.ceil(idx)
  return lo === hi ? sorted[lo] : sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo)
}
function stats (values) {
  const sorted = [...values].sort((a, b) => a - b)
  const mean = sorted.reduce((a, b) => a + b, 0) / sorted.length
  const variance = sorted.reduce((s, v) => s + (v - mean) ** 2, 0) / sorted.length
  return { mean, min: sorted[0], max: sorted[sorted.length - 1], stddev: Math.sqrt(variance), p50: percentile(sorted, 50), p95: percentile(sorted, 95), count: sorted.length }
}
function normWords (text) {
  return (text || '').toLowerCase().replace(/[^\p{L}\p{N}]+/gu, ' ').trim().split(/\s+/).filter(Boolean)
}
function werCounts (refText, hypText) {
  const r = normWords(refText); const h = normWords(hypText)
  if (r.length === 0) return { edits: 0, refLen: 0 }
  const d = Array.from({ length: r.length + 1 }, () => new Array(h.length + 1).fill(0))
  for (let i = 0; i <= r.length; i++) d[i][0] = i
  for (let j = 0; j <= h.length; j++) d[0][j] = j
  for (let i = 1; i <= r.length; i++) {
    for (let j = 1; j <= h.length; j++) {
      const cost = r[i - 1] === h[j - 1] ? 0 : 1
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    }
  }
  return { edits: d[r.length][h.length], refLen: r.length }
}
function wer (refText, hypText) {
  const { edits, refLen } = werCounts(refText, hypText)
  return refLen === 0 ? null : edits / refLen
}

// ---------------------------------------------------------------------------
// engine runners
// ---------------------------------------------------------------------------
// qvac's parakeet-cpp engine, one `--bench` invocation per clip. We read the
// engine-only `inference_ms.samples` (mel + encoder + decoder; excludes model
// load + wav read) so it matches mudler's engine-only `bench`.
function runQvac (modelType, useGPU) {
  const backend = useGPU ? 'gpu' : 'cpu'
  console.log(`[qvac] ${modelType} ${backend} (parakeet-cpp engine --bench)`)
  const cfg = MODEL_MAP[modelType]
  const byClip = {}
  let backendId = null
  for (const c of CLIPS) {
    const out = path.join(OUT_DIR, `qvac-${modelType}-${backend}-${c.id}.json`)
    const args = ['--model', path.join(QVAC_MODELS, cfg.qvac), '--wav', c.wav,
      '--bench', '--bench-runs', String(RUNS), '--bench-warmup', String(WARMUP),
      '--bench-json', out, '--threads', String(THREADS),
      '--n-gpu-layers', useGPU ? '1' : '0']
    const res = spawnSync(QVAC_CLI, args, { cwd: PKG_DIR, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024, env: process.env })
    if (res.status !== 0 || !fs.existsSync(out)) {
      console.error((res.stderr || '').slice(-2000))
      throw new Error(`qvac engine bench failed for ${modelType} ${backend} ${c.id}`)
    }
    const j = JSON.parse(fs.readFileSync(out, 'utf8'))
    backendId = j.backend
    byClip[`${c.id}.wav`] = { proc: stats(j.inference_ms.samples), text: j.transcript, audioSec: j.audio_seconds }
  }
  return { backendId, byClip }
}

function runMudler (modelType, useGPU) {
  const backend = useGPU ? 'gpu' : 'cpu'
  console.log(`[mudler] ${modelType} ${backend}`)
  const cfg = MODEL_MAP[modelType]
  const manifest = path.join(OUT_DIR, `manifest-${modelType}-${backend}.txt`)
  // Each clip repeated RUNS times for steady-state reps (bench warms once).
  const lines = []
  for (const c of CLIPS) for (let i = 0; i < RUNS; i++) lines.push(c.wav)
  fs.writeFileSync(manifest, lines.join('\n') + '\n')
  const jsonOut = path.join(OUT_DIR, `mudler-${modelType}-${backend}.json`)
  const args = ['bench', '--model', path.join(MUDLER_MODELS, cfg.mudler), '--manifest', manifest, '--threads', String(THREADS), '--json', jsonOut]
  if (cfg.decoder) args.push('--decoder', cfg.decoder)
  const env = { ...process.env }
  if (!useGPU) env.PARAKEET_DEVICE = 'cpu'
  const res = spawnSync(MUDLER_CLI, args, { cwd: MUDLER_DIR, env, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 })
  if (res.status !== 0) { console.error(res.stderr || res.stdout); throw new Error(`mudler bench failed for ${modelType} ${backend}`) }
  const j = JSON.parse(fs.readFileSync(jsonOut, 'utf8'))
  const byClip = {}
  for (const f of j.files) {
    const key = path.basename(f.path)
    if (!byClip[key]) byClip[key] = { procVals: [], text: f.text, audioSec: f.audio_sec }
    byClip[key].procVals.push(f.proc_ms)
    byClip[key].text = f.text
  }
  for (const k of Object.keys(byClip)) { byClip[k].proc = stats(byClip[k].procVals); delete byClip[k].procVals }
  return { loadMs: j.load_ms, byClip }
}

// ---------------------------------------------------------------------------
// FLEURS multilingual accuracy (real ground-truth WER, TDT only)
// ---------------------------------------------------------------------------
const FLEURS_MANIFEST = path.join(OUT_DIR, 'fleurs', 'manifest.json')

function runFleurs () {
  if (!fs.existsSync(FLEURS_MANIFEST)) {
    console.log('FLEURS manifest not found; skipping multilingual WER (run fetch-fleurs.js first)')
    return null
  }
  const manifest = JSON.parse(fs.readFileSync(FLEURS_MANIFEST, 'utf8'))
  const wavs = manifest.map(m => m.wav)
  const useGPU = process.env.QVAC_CMP_FLEURS_GPU !== 'false' // GPU by default
  const backend = useGPU ? 'gpu' : 'cpu'
  console.log(`[fleurs] TDT ${backend} over ${manifest.length} utterances`)

  // qvac parakeet-cpp engine: one bench (1 run) per utterance, transcript only.
  const qDir = path.join(OUT_DIR, `fleurs-qvac-${backend}`)
  if (!fs.existsSync(qDir)) fs.mkdirSync(qDir, { recursive: true })
  const qText = {}
  for (const w of wavs) {
    const key = path.basename(w)
    const out = path.join(qDir, key + '.json')
    const qArgs = ['--model', path.join(QVAC_MODELS, MODEL_MAP.tdt.qvac), '--wav', w,
      '--bench', '--bench-runs', '1', '--bench-warmup', '0', '--bench-json', out,
      '--threads', String(THREADS), '--n-gpu-layers', useGPU ? '1' : '0']
    const qr = spawnSync(QVAC_CLI, qArgs, { cwd: PKG_DIR, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024, env: process.env })
    if (qr.status !== 0 || !fs.existsSync(out)) { console.error((qr.stderr || '').slice(-2000)); throw new Error(`fleurs qvac run failed for ${key}`) }
    qText[key] = JSON.parse(fs.readFileSync(out, 'utf8')).transcript
  }

  // mudler (one bench pass, each clip once)
  const manPath = path.join(OUT_DIR, `fleurs-manifest-${backend}.txt`)
  fs.writeFileSync(manPath, wavs.join('\n') + '\n')
  const mOut = path.join(OUT_DIR, `fleurs-mudler-${backend}.json`)
  const mArgs = ['bench', '--model', path.join(MUDLER_MODELS, MODEL_MAP.tdt.mudler), '--manifest', manPath, '--threads', String(THREADS), '--decoder', 'tdt', '--json', mOut]
  const env = { ...process.env }
  if (!useGPU) env.PARAKEET_DEVICE = 'cpu'
  const mr = spawnSync(MUDLER_CLI, mArgs, { cwd: MUDLER_DIR, env, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 })
  if (mr.status !== 0) { console.error(mr.stderr || mr.stdout); throw new Error('fleurs mudler bench failed') }
  const mText = {}
  for (const f of JSON.parse(fs.readFileSync(mOut, 'utf8')).files) mText[path.basename(f.path)] = f.text

  // Corpus WER per language (sum edits / sum ref words).
  const agg = {}
  const detail = []
  for (const m of manifest) {
    const key = path.basename(m.wav)
    const q = werCounts(m.reference, qText[key] || '')
    const mu = werCounts(m.reference, mText[key] || '')
    const a = (agg[m.lang] = agg[m.lang] || { name: m.name, n: 0, refWords: 0, qEdits: 0, mEdits: 0 })
    a.n++; a.refWords += q.refLen; a.qEdits += q.edits; a.mEdits += mu.edits
    detail.push({ id: m.id, lang: m.lang, ref: m.reference, qvac: qText[key] || '', mudler: mText[key] || '' })
  }
  const perLang = Object.entries(agg).map(([lang, a]) => ({
    lang, name: a.name, n: a.n, refWords: a.refWords,
    qvacWer: a.refWords ? a.qEdits / a.refWords : null,
    mudlerWer: a.refWords ? a.mEdits / a.refWords : null
  }))
  return { backend, perLang, detail }
}

// ---------------------------------------------------------------------------
// matrix
// ---------------------------------------------------------------------------
function buildMatrix () {
  const rows = []
  for (const modelType of MODELS) {
    if (!MODEL_MAP[modelType]) { console.warn(`skip unknown ${modelType}`); continue }
    for (const useGPU of GPU_MODES) {
      const q = runQvac(modelType, useGPU)
      const m = runMudler(modelType, useGPU)
      const clips = CLIPS.map(c => {
        const key = `${c.id}.wav`
        const qc = q.byClip[key]; const mc = m.byClip[key]
        const qRtf = (qc.proc.mean / 1000) / qc.audioSec
        const mRtf = (mc.proc.mean / 1000) / mc.audioSec
        // WER applicability: English clips for any model; non-English only for
        // the multilingual TDT model. Reference WER where we have ground truth,
        // else cross-engine agreement.
        const langOk = c.lang === 'en' || MODEL_MAP[modelType].multilingual
        const out = {
          id: c.id, lang: c.lang, audioSec: c.audioSec,
          qvac: { procMs: qc.proc, rtf: qRtf, text: qc.text },
          mudler: { procMs: mc.proc, rtf: mRtf, text: mc.text },
          werRefQvac: (langOk && c.ref) ? wer(c.ref, qc.text) : null,
          werRefMudler: (langOk && c.ref) ? wer(c.ref, mc.text) : null,
          werAgreement: langOk ? wer(qc.text, mc.text) : null
        }
        return out
      })
      rows.push({ modelType, label: MODEL_MAP[modelType].label, useGPU, backendId: q.backendId, mudlerLoadMs: m.loadMs, clips })
    }
  }
  return rows
}

// ---------------------------------------------------------------------------
// reports
// ---------------------------------------------------------------------------
function fmt (n, d = 4) { return (n === null || n === undefined) ? 'n/a' : Number(n).toFixed(d) }
function pct (n, d = 1) { return (n === null || n === undefined) ? 'n/a' : (n * 100).toFixed(d) + '%' }
function writeReports (rows, fleurs, existingMeta) {
  const meta = existingMeta || { generatedAt: new Date().toISOString(), platform: `${process.platform}-${process.arch}`, runs: RUNS, warmup: WARMUP, threads: THREADS, quant: QUANT, clips: CLIPS.map(c => ({ id: c.id, lang: c.lang, audioSec: c.audioSec, hasRef: !!c.ref })) }
  fs.writeFileSync(path.join(OUT_DIR, `comparison-data-${PLATFORM_SLUG}.json`), JSON.stringify({ meta, rows, fleurs }, null, 2) + '\n')
  fs.writeFileSync(path.join(OUT_DIR, `report-${PLATFORM_SLUG}.md`), renderMd(meta, rows, fleurs))
  fs.writeFileSync(path.join(OUT_DIR, `report-${PLATFORM_SLUG}.html`), renderHtml(meta, rows, fleurs))
}

const CANON = 'alice' // headline clip (20s English), matches earlier runs

function renderMd (meta, rows, fleurs) {
  const L = []
  L.push('# Parakeet Engine Comparison: qvac vs mudler/parakeet.cpp')
  L.push('')
  L.push(`Generated: ${meta.generatedAt}  `)
  L.push(`Platform: \`${meta.platform}\` (${PLATFORM_NOTE})  `)
  L.push(`Quant: \`${meta.quant}\` · Threads: ${meta.threads} · Warmup: ${meta.warmup} · Timed reps: ${meta.runs}`)
  L.push('')
  L.push('**RTF** = proc/audio (lower is faster) · **WER** lower is better.')
  L.push('')
  L.push('> Both timings are **engine-only C++ inference** (mel + encoder + decoder), excluding model load and wav read — qvac = `parakeet-cpp --bench` (`inference_ms`), mudler = `parakeet-cli bench` (`transcribe_pcm`). Same canonical clips, same threads, same quant level. Each engine loads its own native q8_0 GGUF (the two schemas are not interchangeable).')
  L.push('>')
  L.push('> **Benchmarked binaries:** the qvac side is the standalone `parakeet-cpp` engine CLI from `tetherto/qvac-ext-lib-whisper.cpp` — **not** the Bare/Node `transcription-parakeet` addon — so no JS/Bare runtime overhead is included. This is an engine-to-engine (C++ vs C++) comparison.')
  L.push('')
  L.push('## Model types in this benchmark')
  L.push('')
  L.push('New to Parakeet? These are the three ASR "decoders" compared here (all share the same FastConformer audio encoder; they differ in how they turn encoder output into text).')
  L.push('')
  L.push('| Model | Full name | How it works | Trade-off | Languages |')
  L.push('|-------|-----------|--------------|-----------|-----------|')
  L.push('| **CTC** | Connectionist Temporal Classification | Non-autoregressive: predicts one token per audio frame (plus a "blank"), then collapses repeats/blanks into text in a single pass. | Fastest & simplest; no explicit duration model, slightly weaker on hard audio. | English |')
  L.push('| **TDT** | Token-and-Duration Transducer (RNN-T family) | A transducer that predicts each token *and how many frames to skip* (its duration), striding over audio instead of stepping frame-by-frame. | Best accuracy + punctuation/capitalization; multilingual. Slightly heavier decoder. | ~25 (v3) |')
  L.push('| **EOU** | End-of-Utterance streaming (RNN-T + `<EOU>`) | A small 120M streaming model that also emits an `<EOU>` token to detect when a speaker finished their turn. | Built for low-latency live conversation / turn-taking, not peak accuracy. | English |')
  L.push('')
  L.push('> Not benchmarked here: **Sortformer** — speaker *diarization* ("who spoke when"), which is qvac-only.')
  L.push('')
  L.push('## Platform & GPU support matrix')
  L.push('')
  L.push('What each project supports out of the box (CPU is available everywhere; **bold** = GPU acceleration).')
  L.push('')
  L.push('| Platform / Arch | qvac transcription-parakeet | mudler/parakeet.cpp |')
  L.push('|-----------------|-----------------------------|---------------------|')
  L.push('| macOS arm64 | CPU + **Metal** | CPU + **Metal** |')
  L.push('| macOS x64 | CPU + **Metal** | CPU only |')
  L.push('| iOS arm64 | CPU + **Metal** | — not supported |')
  L.push('| Linux x64 | CPU + **Vulkan** | CPU + **Vulkan** + **CUDA** |')
  L.push('| Linux arm64 | CPU + **Vulkan** | CPU only |')
  L.push('| Android arm64 | CPU + **Vulkan / OpenCL** | — not supported |')
  L.push('| Windows x64 | CPU + **Vulkan** | CPU + **Vulkan** + **CUDA** |')
  L.push('| AMD (ROCm/HIP) | — | source build (`PARAKEET_GGML_HIP`) |')
  L.push('')
  L.push('**GPU backends:** Metal (both) · Vulkan (both) · **OpenCL → qvac only** (Android/Adreno) · **CUDA + HIP → mudler only** (NVIDIA / AMD).')
  L.push('')
  L.push(`## 1. Headline speed (clip: ${CANON}, English ~${fmt(meta.clips.find(c => c.id === CANON).audioSec, 1)}s)`)
  L.push('')
  L.push('| Model | Backend | Engine | Proc ms | RTF | Faster |')
  L.push('|-------|---------|--------|--------:|----:|--------|')
  for (const r of rows) {
    const c = r.clips.find(x => x.id === CANON)
    const qf = c.qvac.procMs.mean < c.mudler.procMs.mean
    const ratio = c.mudler.procMs.mean / c.qvac.procMs.mean
    L.push(`| ${r.modelType.toUpperCase()} | ${r.useGPU ? GPU_LABEL : 'CPU'} | qvac | ${fmt(c.qvac.procMs.mean, 1)} | ${fmt(c.qvac.rtf)} | ${qf ? `**qvac** ${fmt(ratio, 2)}x` : ''} |`)
    L.push(`| ${r.modelType.toUpperCase()} | ${r.useGPU ? GPU_LABEL : 'CPU'} | mudler | ${fmt(c.mudler.procMs.mean, 1)} | ${fmt(c.mudler.rtf)} | ${!qf ? `**mudler** ${fmt(1 / ratio, 2)}x` : ''} |`)
  }
  L.push('')
  L.push('## 2. RTF vs clip duration (speed stability across lengths)')
  L.push('')
  for (const r of rows) {
    L.push(`### ${r.modelType.toUpperCase()} — ${r.useGPU ? GPU_LABEL : 'CPU'}`)
    L.push('')
    L.push('| Clip | Lang | Duration s | qvac RTF | mudler RTF | Faster |')
    L.push('|------|------|-----------:|---------:|-----------:|--------|')
    for (const c of r.clips) {
      const qf = c.qvac.rtf < c.mudler.rtf
      const ratio = qf ? c.mudler.rtf / c.qvac.rtf : c.mudler.rtf === 0 ? 0 : c.qvac.rtf / c.mudler.rtf
      L.push(`| ${c.id} | ${c.lang} | ${fmt(c.audioSec, 1)} | ${fmt(c.qvac.rtf)} | ${fmt(c.mudler.rtf)} | ${qf ? 'qvac' : 'mudler'} ${fmt(ratio, 2)}x |`)
    }
    L.push('')
  }
  L.push('## 3. Accuracy (WER)')
  L.push('')
  L.push('Reference WER uses ground-truth transcripts (English clips). Agreement = WER between the two engines (proxy for divergence; 0% = byte-identical word stream). Non-English WER only shown for the multilingual TDT model.')
  L.push('')
  L.push('| Model | Backend | Clip | Lang | qvac WER (ref) | mudler WER (ref) | Agreement |')
  L.push('|-------|---------|------|------|---------------:|-----------------:|----------:|')
  for (const r of rows) {
    for (const c of r.clips) {
      if (c.werRefQvac === null && c.werAgreement === null) continue
      L.push(`| ${r.modelType.toUpperCase()} | ${r.useGPU ? GPU_LABEL : 'CPU'} | ${c.id} | ${c.lang} | ${pct(c.werRefQvac)} | ${pct(c.werRefMudler)} | ${pct(c.werAgreement)} |`)
    }
  }
  L.push('')
  if (fleurs) {
    L.push(`## 4. Multilingual accuracy — FLEURS ground truth (TDT, ${fleurs.backend === 'gpu' ? GPU_LABEL : 'CPU'})`)
    L.push('')
    L.push('Real WER against FLEURS reference transcripts (corpus-level: total word edits / total reference words). TDT 0.6B v3 is the only multilingual model; CTC/EOU are English-only and excluded here.')
    L.push('')
    L.push('| Language | Utterances | Ref words | qvac WER | mudler WER | Closer to reference |')
    L.push('|----------|-----------:|----------:|---------:|-----------:|---------------------|')
    for (const p of fleurs.perLang) {
      const qBetter = p.qvacWer <= p.mudlerWer
      L.push(`| ${p.name} (${p.lang}) | ${p.n} | ${p.refWords} | ${pct(p.qvacWer)} | ${pct(p.mudlerWer)} | ${qBetter ? 'qvac' : 'mudler'} |`)
    }
    L.push('')
  }
  L.push('## Feature differences')
  L.push('')
  L.push('Both are ggml ports of the same NVIDIA Parakeet checkpoints, but they target different products, so their feature sets diverge.')
  L.push('')
  L.push('### Only in qvac (`transcription-parakeet`)')
  L.push('- **Speaker diarization** — Sortformer v1 / v2 / v2.1 with NeMo Audio-Online Speaker Cache (AOSC) so speakers rebind to their slot across gaps. mudler has no diarization at all.')
  L.push('- **Speaker-attributed transcription** ("who said what") — ASR + Sortformer combined into one tagged transcript.')
  L.push('- **Live duplex streaming + microphone** — Mode 3 cache-aware chunks (left-context / right-lookahead), `<EOU>` turn boundaries, `StreamEvent` callbacks, energy VAD, and `live-mic` / `live-mic-attributed` example apps.')
  L.push('- **Mobile & embedded reach** — iOS and Android (arm64) builds, plus the **OpenCL** backend for Adreno GPUs.')
  L.push('- **Runtime integration** — ships as a Bare/Node native addon driven from the QVAC SDK (JS API, P2P, batched `run()` / streaming `runStreaming()`), not just a CLI.')
  L.push('')
  L.push('### Only in mudler (`parakeet.cpp`)')
  L.push('- **CUDA (NVIDIA) and HIP/ROCm (AMD) backends** — qvac is Metal / Vulkan / OpenCL only (no CUDA).')
  L.push('- **K-quants** (`q4_k`, `q5_k`, `q6_k`) via `parakeet-cli quantize`. qvac ships `f16 / q8_0 / q5_0 / q4_0` only.')
  L.push('- **More & larger checkpoints** — 1.1B family (CTC / RNNT / TDT / hybrid TDT+CTC), 110M hybrid, RNNT 0.6B, and **nemotron-3.5 streaming multilingual** (40+ locales, prompt-conditioned, `--lang`).')
  L.push('- **Batched decode** (`bench-batch`, `--batch-sizes`) and a `bench-decode` microbenchmark.')
  L.push('- **Distribution surface** — flat C-API (`parakeet_capi.h`) + shared lib for dlopen/FFI/LocalAI, prebuilt CLI binaries for 5 platforms, and Docker images (CPU + CUDA, multi-arch) on GHCR.')
  L.push('- **Word/segment timestamps** (`--timestamps`).')
  L.push('')
  L.push('### Shared by both')
  L.push('CTC + TDT + EOU transcription · `q8_0` / `f16` · CPU + Metal + Vulkan · ggml-based · log-mel front-end on GPU · WER-0 parity vs NeMo on clean English.')
  L.push('')
  L.push('## Benchmark caveats')
  L.push('- CTC and EOU are English-only; their transcripts on non-English clips are expected to be wrong (timing still valid).')
  L.push('- GGUF schemas are not interchangeable (verified both directions): qvac uses renamed `blk`-style tensors + `parakeet.*` KV; mudler keeps verbatim NeMo names. Each engine runs its own native q8_0 file.')
  L.push('- On NVIDIA the GPU column means **CUDA for mudler / Vulkan for qvac** unless mudler is also built with Vulkan; qvac\'s actual backend is recorded as `backendId` in the JSON.')
  L.push('')
  return L.join('\n')
}

function renderHtml (meta, rows, fleurs) {
  const esc = s => String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]))
  const canonSec = meta.clips.find(c => c.id === CANON).audioSec

  const fleursSection = !fleurs ? '' : `
<h2>4. Multilingual accuracy — FLEURS ground truth (TDT, ${fleurs.backend === 'gpu' ? GPU_LABEL : 'CPU'})</h2>
<div class="meta">Real WER vs FLEURS reference transcripts (corpus-level: total word edits / total reference words). TDT 0.6B v3 is the only multilingual model; CTC/EOU excluded.</div>
<table><thead><tr><th>Language</th><th>Utts</th><th>Ref words</th><th>qvac WER</th><th>mudler WER</th><th>Closer to ref</th></tr></thead><tbody>${
  fleurs.perLang.map(p => {
    const qBetter = p.qvacWer <= p.mudlerWer
    return `<tr><td style="text-align:left">${esc(p.name)} (${p.lang})</td><td>${p.n}</td><td>${p.refWords}</td><td class="${qBetter ? 'win' : ''}">${pct(p.qvacWer)}</td><td class="${!qBetter ? 'win' : ''}">${pct(p.mudlerWer)}</td><td style="text-align:left">${qBetter ? 'qvac' : 'mudler'}</td></tr>`
  }).join('')
}</tbody></table>`

  const headlineRows = rows.map(r => {
    const c = r.clips.find(x => x.id === CANON)
    const qf = c.qvac.procMs.mean < c.mudler.procMs.mean
    const ratio = c.mudler.procMs.mean / c.qvac.procMs.mean
    const winText = qf ? `qvac ${ratio.toFixed(2)}\u00d7` : `mudler ${(1 / ratio).toFixed(2)}\u00d7`
    const qCls = qf ? 'win' : ''; const mCls = !qf ? 'win' : ''
    return `<tr class="grp"><td rowspan="2">${r.modelType.toUpperCase()}</td><td rowspan="2">${r.useGPU ? GPU_LABEL : 'CPU'}</td>
      <td class="eng qvac">qvac</td><td class="${qCls}">${c.qvac.procMs.mean.toFixed(1)}</td><td>${c.qvac.rtf.toFixed(4)}</td><td rowspan="2">${winText} faster</td></tr>
      <tr><td class="eng mudler">mudler</td><td class="${mCls}">${c.mudler.procMs.mean.toFixed(1)}</td><td>${c.mudler.rtf.toFixed(4)}</td></tr>`
  }).join('')

  // RTF-vs-duration line chart (one panel per model, both backends, both engines).
  const chart = renderRtfChart(rows)

  const durRows = rows.map(r => {
    const head = `<tr class="section"><td colspan="6">${r.modelType.toUpperCase()} — ${r.useGPU ? GPU_LABEL : 'CPU'}</td></tr>`
    const body = r.clips.map(c => {
      const qf = c.qvac.rtf < c.mudler.rtf
      const ratio = qf ? c.mudler.rtf / c.qvac.rtf : (c.mudler.rtf === 0 ? 0 : c.qvac.rtf / c.mudler.rtf)
      return `<tr><td>${c.id}</td><td>${c.lang}</td><td>${c.audioSec.toFixed(1)}</td><td class="${qf ? 'win' : ''}">${c.qvac.rtf.toFixed(4)}</td><td class="${!qf ? 'win' : ''}">${c.mudler.rtf.toFixed(4)}</td><td style="text-align:left">${qf ? 'qvac' : 'mudler'} ${ratio.toFixed(2)}\u00d7</td></tr>`
    }).join('')
    return head + body
  }).join('')

  const accRows = rows.flatMap(r => r.clips.filter(c => c.werRefQvac !== null || c.werAgreement !== null).map(c => {
    const flag = (c.werRefQvac !== null && c.werRefQvac > 0.3) ? ' class="warn"' : ''
    return `<tr${flag}><td>${r.modelType.toUpperCase()}</td><td>${r.useGPU ? GPU_LABEL : 'CPU'}</td><td>${c.id}</td><td>${c.lang}</td><td>${pct(c.werRefQvac)}</td><td>${pct(c.werRefMudler)}</td><td>${pct(c.werAgreement)}</td></tr>`
  })).join('')

  return `<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Parakeet: qvac vs mudler/parakeet.cpp</title>
<style>
:root { color-scheme: light dark; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 2rem; background: #0d1117; color: #e6edf3; }
h1 { font-size: 1.5rem; margin: 0 0 .25rem; } h2 { font-size: 1.15rem; margin: 2rem 0 .5rem; border-bottom: 1px solid #21262d; padding-bottom: .35rem; }
.meta { color: #8b949e; font-size: .85rem; line-height: 1.7; margin-bottom: 1rem; } .meta code { background: #161b22; padding: .1rem .35rem; border-radius: 4px; }
table { border-collapse: collapse; width: 100%; margin: .5rem 0 1.5rem; font-size: .88rem; }
th, td { padding: .45rem .7rem; text-align: right; border-bottom: 1px solid #21262d; }
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), .eng { text-align: left; }
thead th { background: #161b22; color: #8b949e; font-weight: 600; }
tr.grp td, tr.section td { border-top: 2px solid #30363d; }
tr.section td { text-align: left; color: #58a6ff; font-weight: 700; background: #161b22; }
.eng.qvac { color: #58a6ff; font-weight: 600; } .eng.mudler { color: #f0883e; font-weight: 600; }
td.win { background: rgba(63,185,80,.15); font-weight: 700; } tr.warn td { background: rgba(248,81,73,.12); }
.legend span { display:inline-block; margin-right:1rem; font-size:.85rem; } .sw { display:inline-block; width:12px; height:12px; border-radius:2px; vertical-align:middle; margin-right:.35rem; }
.bq { fill:#58a6ff; } .bm { fill:#f0883e; } svg { background:#0d1117; border:1px solid #21262d; border-radius:8px; }
.note { font-size:.85rem; color:#8b949e; max-width:78ch; line-height:1.7; } .note li { margin:.3rem 0; }
.grid { display:flex; flex-wrap:wrap; gap:1rem; }
</style></head><body>
<h1>Parakeet Engine Comparison</h1>
<div class="meta">
qvac <code>parakeet-cpp</code> engine CLI vs <code>mudler/parakeet.cpp</code> (parakeet-cli)<br/>
Generated ${esc(meta.generatedAt)} · Platform <code>${esc(meta.platform)}</code> (${esc(PLATFORM_NOTE)})<br/>
Quant <code>${esc(meta.quant)}</code> · Threads ${meta.threads} · Warmup ${meta.warmup} · Timed reps ${meta.runs}<br/>
<strong>RTF</strong> = proc/audio (lower is faster) · <strong>WER</strong> lower is better<br/>
Both timings are <strong>engine-only C++ inference</strong> (mel + encoder + decoder), excluding model load + wav read — qvac = <code>parakeet-cpp --bench</code> (inference_ms), mudler = <code>parakeet-cli bench</code> (transcribe_pcm).<br/>
<strong>Benchmarked binaries:</strong> the qvac side is the standalone <code>parakeet-cpp</code> engine CLI (from <code>tetherto/qvac-ext-lib-whisper.cpp</code>) — <strong>not</strong> the Bare/Node <code>transcription-parakeet</code> addon — so no JS/Bare overhead is included (engine-to-engine, C++ vs C++).
</div>
<h2>Model types in this benchmark</h2>
<div class="meta">All three share the same FastConformer audio encoder; they differ in how they turn encoder output into text. Sortformer (speaker diarization) is qvac-only and not benchmarked here.</div>
<table><thead><tr><th>Model</th><th>Full name</th><th>How it works</th><th>Trade-off</th><th>Langs</th></tr></thead><tbody>
<tr><td><strong>CTC</strong></td><td style="text-align:left">Connectionist Temporal Classification</td><td style="text-align:left">Non-autoregressive: one token per audio frame (plus "blank"), collapsed into text in a single pass.</td><td style="text-align:left">Fastest &amp; simplest; no duration model, slightly weaker on hard audio.</td><td>English</td></tr>
<tr><td><strong>TDT</strong></td><td style="text-align:left">Token-and-Duration Transducer (RNN-T family)</td><td style="text-align:left">Predicts each token <em>and how many frames to skip</em>, striding over audio instead of frame-by-frame.</td><td style="text-align:left">Best accuracy + punctuation/caps; multilingual. Heavier decoder.</td><td>~25 (v3)</td></tr>
<tr><td><strong>EOU</strong></td><td style="text-align:left">End-of-Utterance streaming (RNN-T + &lt;EOU&gt;)</td><td style="text-align:left">Small 120M streaming model that also emits an &lt;EOU&gt; token to detect end-of-turn.</td><td style="text-align:left">Built for low-latency live conversation, not peak accuracy.</td><td>English</td></tr>
</tbody></table>
<h2>Platform &amp; GPU support matrix</h2>
<div class="meta">What each project supports out of the box. CPU is available everywhere; <strong>bold</strong> = GPU acceleration.</div>
<table><thead><tr><th>Platform / Arch</th><th>qvac transcription-parakeet</th><th>mudler/parakeet.cpp</th></tr></thead><tbody>
<tr><td style="text-align:left">macOS arm64</td><td style="text-align:left">CPU + <strong>Metal</strong></td><td style="text-align:left">CPU + <strong>Metal</strong></td></tr>
<tr><td style="text-align:left">macOS x64</td><td style="text-align:left">CPU + <strong>Metal</strong></td><td style="text-align:left">CPU only</td></tr>
<tr><td style="text-align:left">iOS arm64</td><td style="text-align:left">CPU + <strong>Metal</strong></td><td style="text-align:left">— not supported</td></tr>
<tr><td style="text-align:left">Linux x64</td><td style="text-align:left">CPU + <strong>Vulkan</strong></td><td style="text-align:left">CPU + <strong>Vulkan</strong> + <strong>CUDA</strong></td></tr>
<tr><td style="text-align:left">Linux arm64</td><td style="text-align:left">CPU + <strong>Vulkan</strong></td><td style="text-align:left">CPU only</td></tr>
<tr><td style="text-align:left">Android arm64</td><td style="text-align:left">CPU + <strong>Vulkan / OpenCL</strong></td><td style="text-align:left">— not supported</td></tr>
<tr><td style="text-align:left">Windows x64</td><td style="text-align:left">CPU + <strong>Vulkan</strong></td><td style="text-align:left">CPU + <strong>Vulkan</strong> + <strong>CUDA</strong></td></tr>
<tr><td style="text-align:left">AMD (ROCm/HIP)</td><td style="text-align:left">—</td><td style="text-align:left">source build (PARAKEET_GGML_HIP)</td></tr>
</tbody></table>
<div class="meta"><strong>GPU backends:</strong> Metal (both) · Vulkan (both) · <strong>OpenCL → qvac only</strong> (Android/Adreno) · <strong>CUDA + HIP → mudler only</strong> (NVIDIA / AMD).</div>
<h2>1. Headline speed — clip <code>${CANON}</code> (English ~${canonSec.toFixed(1)}s)</h2>
<table><thead><tr><th>Model</th><th>Backend</th><th>Engine</th><th>Proc ms</th><th>RTF</th><th>Faster</th></tr></thead><tbody>${headlineRows}</tbody></table>
<h2>2. RTF vs clip duration</h2>
<div class="legend"><span><span class="sw bq"></span>qvac</span><span><span class="sw bm"></span>mudler</span><span>solid = CPU, dashed = ${esc(GPU_LABEL)} · lower RTF = faster</span></div>
<div class="grid">${chart}</div>
<table><thead><tr><th>Clip</th><th>Lang</th><th>Dur s</th><th>qvac RTF</th><th>mudler RTF</th><th>Faster</th></tr></thead><tbody>${durRows}</tbody></table>
<h2>3. Accuracy (WER)</h2>
<div class="meta">Reference WER = vs ground-truth transcript (English clips). Agreement = WER between the two engines (0% means identical word stream). Non-English WER shown only for multilingual TDT. Rows with qvac reference WER &gt; 30% are highlighted.</div>
<table><thead><tr><th>Model</th><th>Backend</th><th>Clip</th><th>Lang</th><th>qvac WER (ref)</th><th>mudler WER (ref)</th><th>Agreement</th></tr></thead><tbody>${accRows}</tbody></table>
${fleursSection}
<h2>Feature differences</h2>
<div class="meta">Both are ggml ports of the same NVIDIA Parakeet checkpoints, but they target different products, so feature sets diverge.</div>
<p class="note"><strong style="color:#58a6ff">Only in qvac (transcription-parakeet)</strong></p>
<ul class="note">
<li><strong>Speaker diarization</strong> — Sortformer v1 / v2 / v2.1 with NeMo Audio-Online Speaker Cache (AOSC). mudler has no diarization at all.</li>
<li><strong>Speaker-attributed transcription</strong> ("who said what") — ASR + Sortformer combined into one tagged transcript.</li>
<li><strong>Live duplex streaming + microphone</strong> — Mode 3 cache-aware chunks (left-context / right-lookahead), &lt;EOU&gt; turn boundaries, StreamEvent callbacks, energy VAD, live-mic example apps.</li>
<li><strong>Mobile &amp; embedded reach</strong> — iOS and Android (arm64) builds, plus the <strong>OpenCL</strong> backend for Adreno GPUs.</li>
<li><strong>Runtime integration</strong> — ships as a Bare/Node native addon driven from the QVAC SDK (JS API, P2P), not just a CLI.</li>
</ul>
<p class="note"><strong style="color:#f0883e">Only in mudler (parakeet.cpp)</strong></p>
<ul class="note">
<li><strong>CUDA (NVIDIA) and HIP/ROCm (AMD) backends</strong> — qvac is Metal / Vulkan / OpenCL only (no CUDA).</li>
<li><strong>K-quants</strong> (q4_k, q5_k, q6_k) via <code>parakeet-cli quantize</code>. qvac ships f16 / q8_0 / q5_0 / q4_0 only.</li>
<li><strong>More &amp; larger checkpoints</strong> — 1.1B family (CTC/RNNT/TDT/hybrid TDT+CTC), 110M hybrid, RNNT 0.6B, and nemotron-3.5 streaming multilingual (40+ locales, prompt-conditioned).</li>
<li><strong>Batched decode</strong> (bench-batch, --batch-sizes) and a bench-decode microbenchmark.</li>
<li><strong>Distribution surface</strong> — flat C-API (parakeet_capi.h) + shared lib for dlopen/FFI/LocalAI, prebuilt CLI binaries for 5 platforms, and Docker images (CPU + CUDA) on GHCR.</li>
<li><strong>Word/segment timestamps</strong> (--timestamps).</li>
</ul>
<p class="note"><strong>Shared by both</strong></p>
<ul class="note">
<li>CTC + TDT + EOU transcription · q8_0 / f16 · CPU + Metal + Vulkan · ggml-based · log-mel front-end on GPU · WER-0 parity vs NeMo on clean English.</li>
</ul>
<h2>Benchmark caveats</h2>
<ul class="note">
<li>CTC &amp; EOU are English-only; their non-English transcripts are expected to be wrong (timing still valid).</li>
<li>GGUF schemas are not interchangeable (verified both ways): qvac renamed <code>blk</code> tensors + <code>parakeet.*</code> KV; mudler verbatim NeMo names.</li>
<li>On NVIDIA the GPU column means <strong>CUDA for mudler / Vulkan for qvac</strong> unless mudler is also built with Vulkan; qvac's actual backend is the <code>backendId</code> in the JSON.</li>
</ul>
</body></html>
`
}

function renderRtfChart (rows) {
  // One SVG panel per model; x = clip duration, y = RTF; qvac vs mudler, CPU solid / Metal dashed.
  const byModel = {}
  for (const r of rows) { (byModel[r.modelType] = byModel[r.modelType] || []).push(r) }
  const W = 300; const H = 200; const padL = 44; const padB = 28; const padT = 12; const padR = 10
  return Object.keys(byModel).map(modelType => {
    const variants = byModel[modelType]
    const pts = variants.flatMap(v => v.clips.flatMap(c => [c.qvac.rtf, c.mudler.rtf]))
    const maxRtf = Math.max(...pts, 0.001) * 1.1
    const durs = variants[0].clips.map(c => c.audioSec)
    const maxDur = Math.max(...durs) * 1.05
    const xx = d => padL + (d / maxDur) * (W - padL - padR)
    const yy = v => (H - padB) - (v / maxRtf) * (H - padB - padT)
    const line = (clips, key, dash) => {
      const ordered = [...clips].sort((a, b) => a.audioSec - b.audioSec)
      const d = ordered.map((c, i) => `${i ? 'L' : 'M'}${xx(c.audioSec).toFixed(1)},${yy(c[key].rtf).toFixed(1)}`).join(' ')
      const cls = key === 'qvac' ? '#58a6ff' : '#f0883e'
      const dots = ordered.map(c => `<circle cx="${xx(c.audioSec).toFixed(1)}" cy="${yy(c[key].rtf).toFixed(1)}" r="2.5" fill="${cls}"/>`).join('')
      return `<path d="${d}" fill="none" stroke="${cls}" stroke-width="1.6" ${dash ? 'stroke-dasharray="4,3"' : ''}/>${dots}`
    }
    const series = variants.map(v => line(v.clips, 'qvac', v.useGPU) + line(v.clips, 'mudler', v.useGPU)).join('')
    const yticks = [0, maxRtf / 2, maxRtf].map(v => `<text x="${padL - 6}" y="${(yy(v) + 3).toFixed(1)}" class="ax" text-anchor="end">${v.toFixed(3)}</text><line x1="${padL}" y1="${yy(v).toFixed(1)}" x2="${W - padR}" y2="${yy(v).toFixed(1)}" stroke="#21262d"/>`).join('')
    const xticks = durs.map(d => `<text x="${xx(d).toFixed(1)}" y="${H - padB + 16}" class="ax" text-anchor="middle">${d.toFixed(0)}</text>`).join('')
    return `<div><div style="font-size:.8rem;color:#8b949e;margin-bottom:.2rem">${modelType.toUpperCase()} — RTF vs duration (s)</div>
      <svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}"><style>.ax{fill:#8b949e;font-size:9px;}</style>${yticks}${xticks}${series}</svg></div>`
  }).join('')
}

function main () {
  // Render-only mode: re-render report .md/.html from an existing data JSON
  // (e.g. regenerate a committed linux-x64 report on a Mac after template edits).
  const renderFrom = process.env.QVAC_CMP_RENDER_FROM
  if (renderFrom) {
    const data = JSON.parse(fs.readFileSync(renderFrom, 'utf8'))
    derivePlatform(data.meta && data.meta.platform)
    if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true })
    writeReports(data.rows, data.fleurs, data.meta)
    console.log(`Re-rendered report-${PLATFORM_SLUG}.{md,html} from ${path.basename(renderFrom)}`)
    return
  }
  console.log('Parakeet comparison harness (engine-only, multi-clip + WER)')
  console.log(`  models: ${MODELS.join(', ')} · gpu: ${GPU_MODES.join(', ')} · runs ${RUNS} (warmup ${WARMUP}) · threads ${THREADS}`)
  console.log(`  qvac engine: ${QVAC_CLI}`)
  console.log(`  mudler engine: ${MUDLER_CLI}`)
  if (!fs.existsSync(QVAC_CLI)) throw new Error(`qvac parakeet-cpp CLI not found at ${QVAC_CLI} (set QVAC_PARAKEET_CLI or build it — see README)`)
  if (!fs.existsSync(MUDLER_CLI)) throw new Error(`mudler parakeet-cli not found at ${MUDLER_CLI}`)
  if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true })
  buildClips()
  console.log('  clips: ' + CLIPS.map(c => `${c.id}(${c.audioSec.toFixed(1)}s/${c.lang})`).join(', '))
  let rows
  const dataPath = path.join(OUT_DIR, `comparison-data-${PLATFORM_SLUG}.json`)
  const legacyDataPath = path.join(OUT_DIR, 'comparison-data.json')
  const cachePath = fs.existsSync(dataPath) ? dataPath : legacyDataPath
  if (process.env.QVAC_CMP_SKIP_MATRIX === '1' && fs.existsSync(cachePath)) {
    console.log(`  (QVAC_CMP_SKIP_MATRIX=1) reusing existing RTF/agreement matrix from ${path.basename(cachePath)}`)
    rows = JSON.parse(fs.readFileSync(cachePath, 'utf8')).rows
  } else {
    rows = buildMatrix()
  }
  const fleurs = runFleurs()
  writeReports(rows, fleurs)
  console.log(`\nReports written to ${OUT_DIR}: report-${PLATFORM_SLUG}.md, report-${PLATFORM_SLUG}.html, comparison-data-${PLATFORM_SLUG}.json`)
}

main()
