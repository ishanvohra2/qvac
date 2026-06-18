#!/usr/bin/env node
'use strict'

/**
 * Fetch a small labelled FLEURS subset (audio + reference transcripts) for the
 * languages supported by parakeet-tdt-0.6b-v3, so the WER section can show real
 * ground-truth WER instead of cross-engine agreement only.
 *
 * By default it covers all 25 TDT v3 languages. For each language it downloads
 * the smallest split (`dev`): the `dev.tsv` (references) and `dev.tar.gz`
 * (16 kHz mono wav audio), extracts the first N utterances, transcodes them to
 * canonical 16 kHz mono PCM16 wav, and writes out/fleurs/manifest.json consumed
 * by run-comparison.js.
 *
 * NOTE: all 25 languages is ~4-5 GB of downloads (each dev.tar.gz is
 * ~150-200 MB). Use FLEURS_LANGS to restrict to a subset of `lang` codes.
 *
 * Usage:
 *   node fetch-fleurs.js [N]                 # default N=12 utts/lang, all 25 langs
 *   FLEURS_LANGS=fr,es,hr node fetch-fleurs.js   # only these languages
 */

const fs = require('fs')
const path = require('path')
const { spawnSync } = require('child_process')

const OUT = path.join(__dirname, 'out', 'fleurs')
const CLIPS = path.join(OUT, 'clips')
const SAMPLE_RATE = 16000
const N = Number.parseInt(process.argv[2] || '12', 10)

// FLEURS config -> human language. transcription column (index 3) is the
// normalized lowercase/no-punct reference, which matches our WER normaliser.
//
// This is the full set of 25 languages supported by parakeet-tdt-0.6b-v3
// (NVIDIA's "25 European languages"). All cfg codes are verified to exist in
// the google/fleurs dataset. Downloading every language's dev split is heavy
// (~150-200 MB tar each, ~4-5 GB total); set FLEURS_LANGS to a comma-separated
// subset of `lang` codes (e.g. FLEURS_LANGS=fr,es,hr) to fetch only some.
const ALL_LANGS = [
  { cfg: 'bg_bg', lang: 'bg', name: 'Bulgarian' },
  { cfg: 'hr_hr', lang: 'hr', name: 'Croatian' },
  { cfg: 'cs_cz', lang: 'cs', name: 'Czech' },
  { cfg: 'da_dk', lang: 'da', name: 'Danish' },
  { cfg: 'nl_nl', lang: 'nl', name: 'Dutch' },
  { cfg: 'en_us', lang: 'en', name: 'English' },
  { cfg: 'et_ee', lang: 'et', name: 'Estonian' },
  { cfg: 'fi_fi', lang: 'fi', name: 'Finnish' },
  { cfg: 'fr_fr', lang: 'fr', name: 'French' },
  { cfg: 'de_de', lang: 'de', name: 'German' },
  { cfg: 'el_gr', lang: 'el', name: 'Greek' },
  { cfg: 'hu_hu', lang: 'hu', name: 'Hungarian' },
  { cfg: 'it_it', lang: 'it', name: 'Italian' },
  { cfg: 'lv_lv', lang: 'lv', name: 'Latvian' },
  { cfg: 'lt_lt', lang: 'lt', name: 'Lithuanian' },
  { cfg: 'mt_mt', lang: 'mt', name: 'Maltese' },
  { cfg: 'pl_pl', lang: 'pl', name: 'Polish' },
  { cfg: 'pt_br', lang: 'pt', name: 'Portuguese' },
  { cfg: 'ro_ro', lang: 'ro', name: 'Romanian' },
  { cfg: 'ru_ru', lang: 'ru', name: 'Russian' },
  { cfg: 'sk_sk', lang: 'sk', name: 'Slovak' },
  { cfg: 'sl_si', lang: 'sl', name: 'Slovenian' },
  { cfg: 'es_419', lang: 'es', name: 'Spanish' },
  { cfg: 'sv_se', lang: 'sv', name: 'Swedish' },
  { cfg: 'uk_ua', lang: 'uk', name: 'Ukrainian' }
]

// Optional subset via FLEURS_LANGS=fr,es,hr ; default = all 25.
const LANG_FILTER = (process.env.FLEURS_LANGS || '').split(',').map(s => s.trim()).filter(Boolean)
const LANGS = LANG_FILTER.length ? ALL_LANGS.filter(l => LANG_FILTER.includes(l.lang)) : ALL_LANGS
const BASE = 'https://huggingface.co/datasets/google/fleurs/resolve/main/data'

function curl (url, outPath) {
  const res = spawnSync('curl', ['-sL', '-C', '-', '-o', outPath, url], { encoding: 'utf8', maxBuffer: 1 << 24 })
  if (res.status !== 0) throw new Error(`curl failed for ${url}: ${res.stderr}`)
}

// Reads a WAV (PCM16 or IEEE-float32) and returns canonical mono Int16.
function readWavInt16 (wavPath) {
  const buf = fs.readFileSync(wavPath)
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength)
  let off = 12; let channels = 1; let bits = 16; let fmt = 1; let dataOff = -1; let dataLen = 0
  while (off + 8 <= buf.byteLength) {
    const id = dv.getUint32(off, false); const sz = dv.getUint32(off + 4, true); const body = off + 8
    if (id === 0x666d7420) { fmt = dv.getUint16(body, true); channels = dv.getUint16(body + 2, true); bits = dv.getUint16(body + 14, true) } else if (id === 0x64617461) { dataOff = body; dataLen = sz }
    off = body + sz + (sz & 1)
  }
  if (dataOff < 0) throw new Error(`no data chunk: ${wavPath}`)
  const bytesPerSample = bits / 8
  const frames = (dataLen / bytesPerSample / channels) | 0
  const out = new Int16Array(frames)
  const readSample = (o) => {
    if (fmt === 3 && bits === 32) return Math.max(-32768, Math.min(32767, Math.round(dv.getFloat32(o, true) * 32767)))
    if (fmt === 1 && bits === 16) return dv.getInt16(o, true)
    throw new Error(`unsupported wav (fmt=${fmt}, bits=${bits}): ${wavPath}`)
  }
  for (let f = 0; f < frames; f++) {
    let acc = 0
    for (let c = 0; c < channels; c++) acc += readSample(dataOff + (f * channels + c) * bytesPerSample)
    out[f] = Math.max(-32768, Math.min(32767, Math.round(acc / channels)))
  }
  return out
}

function writeCanonicalWav (int16, outPath) {
  const dataLen = int16.length * 2
  const h = Buffer.alloc(44)
  h.write('RIFF', 0, 'ascii'); h.writeUInt32LE(36 + dataLen, 4); h.write('WAVE', 8, 'ascii')
  h.write('fmt ', 12, 'ascii'); h.writeUInt32LE(16, 16); h.writeUInt16LE(1, 20); h.writeUInt16LE(1, 22)
  h.writeUInt32LE(SAMPLE_RATE, 24); h.writeUInt32LE(SAMPLE_RATE * 2, 28); h.writeUInt16LE(2, 32); h.writeUInt16LE(16, 34)
  h.write('data', 36, 'ascii'); h.writeUInt32LE(dataLen, 40)
  fs.writeFileSync(outPath, Buffer.concat([h, Buffer.from(int16.buffer, int16.byteOffset, dataLen)]))
}

function parseTsv (tsvPath) {
  const map = new Map()
  for (const line of fs.readFileSync(tsvPath, 'utf8').split('\n')) {
    if (!line.trim()) continue
    const cols = line.split('\t')
    if (cols.length < 4) continue
    map.set(cols[1], cols[3]) // filename -> normalized transcription
  }
  return map
}

function main () {
  fs.mkdirSync(CLIPS, { recursive: true })
  const manifest = []
  for (const { cfg, lang, name } of LANGS) {
    console.log(`\n=== ${name} (${cfg}) ===`)
    const tsvPath = path.join(OUT, `${cfg}.dev.tsv`)
    const tarPath = path.join(OUT, `${cfg}.dev.tar.gz`)
    if (!fs.existsSync(tsvPath)) { console.log('  download tsv'); curl(`${BASE}/${cfg}/dev.tsv`, tsvPath) }
    if (!fs.existsSync(tarPath)) { console.log('  download audio tar (~150-200MB)'); curl(`${BASE}/${cfg}/audio/dev.tar.gz`, tarPath) }
    const refs = parseTsv(tsvPath)

    // List tar members, map basename -> member path.
    const list = spawnSync('tar', ['-tzf', tarPath], { encoding: 'utf8', maxBuffer: 1 << 26 })
    if (list.status !== 0) throw new Error(`tar -tzf failed: ${list.stderr}`)
    const members = new Map()
    for (const m of list.stdout.split('\n')) { if (m.endsWith('.wav')) members.set(path.basename(m), m) }

    // Pick the first N utterances that have both a reference and audio.
    const picked = []
    for (const [filename, ref] of refs) {
      if (picked.length >= N) break
      const member = members.get(filename)
      if (member && ref && ref.trim()) picked.push({ filename, ref, member })
    }
    console.log(`  picked ${picked.length} utterances`)

    // Extract just the picked members.
    const extractDir = path.join(OUT, `${cfg}.extract`)
    fs.mkdirSync(extractDir, { recursive: true })
    const ex = spawnSync('tar', ['-xzf', tarPath, '-C', extractDir, ...picked.map(p => p.member)], { encoding: 'utf8' })
    if (ex.status !== 0) throw new Error(`tar extract failed: ${ex.stderr}`)

    picked.forEach((p, i) => {
      const src = path.join(extractDir, p.member)
      const int16 = readWavInt16(src)
      const outWav = path.join(CLIPS, `${lang}_${i}.wav`)
      writeCanonicalWav(int16, outWav)
      manifest.push({ lang, name, id: `${lang}_${i}`, wav: outWav, reference: p.ref, audioSec: int16.length / SAMPLE_RATE })
    })
  }

  fs.writeFileSync(path.join(OUT, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n')
  const byLang = {}
  for (const m of manifest) byLang[m.lang] = (byLang[m.lang] || 0) + 1
  console.log('\nmanifest:', JSON.stringify(byLang), '->', path.join(OUT, 'manifest.json'))
}

main()
