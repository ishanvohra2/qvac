# Parakeet Engine Comparison: qvac vs mudler/parakeet.cpp

Generated: 2026-06-18T07:54:02.873Z  
Platform: `darwin-arm64` (Apple Silicon, Metal)  
Quant: `q8_0` · Threads: 4 · Warmup: 1 · Timed reps: 5

**RTF** = proc/audio (lower is faster) · **WER** lower is better.

> Both timings are **engine-only C++ inference** (mel + encoder + decoder), excluding model load and wav read — qvac = `parakeet-cpp --bench` (`inference_ms`), mudler = `parakeet-cli bench` (`transcribe_pcm`). Same canonical clips, same threads, same quant level. Each engine loads its own native q8_0 GGUF (the two schemas are not interchangeable).
>
> **Benchmarked binaries:** the qvac side is the standalone `parakeet-cpp` engine CLI from `tetherto/qvac-ext-lib-whisper.cpp` — **not** the Bare/Node `transcription-parakeet` addon — so no JS/Bare runtime overhead is included. This is an engine-to-engine (C++ vs C++) comparison.

## Model types in this benchmark

New to Parakeet? These are the three ASR "decoders" compared here (all share the same FastConformer audio encoder; they differ in how they turn encoder output into text).

| Model | Full name | How it works | Trade-off | Languages |
|-------|-----------|--------------|-----------|-----------|
| **CTC** | Connectionist Temporal Classification | Non-autoregressive: predicts one token per audio frame (plus a "blank"), then collapses repeats/blanks into text in a single pass. | Fastest & simplest; no explicit duration model, slightly weaker on hard audio. | English |
| **TDT** | Token-and-Duration Transducer (RNN-T family) | A transducer that predicts each token *and how many frames to skip* (its duration), striding over audio instead of stepping frame-by-frame. | Best accuracy + punctuation/capitalization; multilingual. Slightly heavier decoder. | ~25 (v3) |
| **EOU** | End-of-Utterance streaming (RNN-T + `<EOU>`) | A small 120M streaming model that also emits an `<EOU>` token to detect when a speaker finished their turn. | Built for low-latency live conversation / turn-taking, not peak accuracy. | English |

> Not benchmarked here: **Sortformer** — speaker *diarization* ("who spoke when"), which is qvac-only.

## Platform & GPU support matrix

What each project supports out of the box (CPU is available everywhere; **bold** = GPU acceleration).

| Platform / Arch | qvac transcription-parakeet | mudler/parakeet.cpp |
|-----------------|-----------------------------|---------------------|
| macOS arm64 | CPU + **Metal** | CPU + **Metal** |
| macOS x64 | CPU + **Metal** | CPU only |
| iOS arm64 | CPU + **Metal** | — not supported |
| Linux x64 | CPU + **Vulkan** | CPU + **Vulkan** + **CUDA** |
| Linux arm64 | CPU + **Vulkan** | CPU only |
| Android arm64 | CPU + **Vulkan / OpenCL** | — not supported |
| Windows x64 | CPU + **Vulkan** | CPU + **Vulkan** + **CUDA** |
| AMD (ROCm/HIP) | — | source build (`PARAKEET_GGML_HIP`) |

**GPU backends:** Metal (both) · Vulkan (both) · **OpenCL → qvac only** (Android/Adreno) · **CUDA + HIP → mudler only** (NVIDIA / AMD).

## 1. Headline speed (clip: alice, English ~20.1s)

| Model | Backend | Engine | Proc ms | RTF | Faster |
|-------|---------|--------|--------:|----:|--------|
| TDT | CPU | qvac | 1652.6 | 0.0821 | **qvac** 1.27x |
| TDT | CPU | mudler | 2095.1 | 0.1041 |  |
| TDT | Metal | qvac | 594.5 | 0.0295 | **qvac** 1.17x |
| TDT | Metal | mudler | 697.1 | 0.0346 |  |
| CTC | CPU | qvac | 2270.2 | 0.1128 | **qvac** 1.30x |
| CTC | CPU | mudler | 2955.9 | 0.1468 |  |
| CTC | Metal | qvac | 521.1 | 0.0259 | **qvac** 1.02x |
| CTC | Metal | mudler | 531.9 | 0.0264 |  |
| EOU | CPU | qvac | 1015.5 | 0.0504 |  |
| EOU | CPU | mudler | 880.0 | 0.0437 | **mudler** 1.15x |
| EOU | Metal | qvac | 407.8 | 0.0203 |  |
| EOU | Metal | mudler | 383.9 | 0.0191 | **mudler** 1.06x |

## 2. RTF vs clip duration (speed stability across lengths)

### TDT — CPU

| Clip | Lang | Duration s | qvac RTF | mudler RTF | Faster |
|------|------|-----------:|---------:|-----------:|--------|
| jfk | en | 11.0 | 0.0920 | 0.0990 | qvac 1.08x |
| alice | en | 20.1 | 0.0821 | 0.1041 | qvac 1.27x |
| croatian | hr | 27.4 | 0.0794 | 0.1044 | qvac 1.31x |
| french | fr | 29.4 | 0.0883 | 0.1149 | qvac 1.30x |
| spanish60 | es | 60.0 | 0.0960 | 0.1384 | qvac 1.44x |

### TDT — Metal

| Clip | Lang | Duration s | qvac RTF | mudler RTF | Faster |
|------|------|-----------:|---------:|-----------:|--------|
| jfk | en | 11.0 | 0.0270 | 0.0287 | qvac 1.06x |
| alice | en | 20.1 | 0.0295 | 0.0346 | qvac 1.17x |
| croatian | hr | 27.4 | 0.0325 | 0.0351 | qvac 1.08x |
| french | fr | 29.4 | 0.0357 | 0.0356 | mudler 1.00x |
| spanish60 | es | 60.0 | 0.0483 | 0.0407 | mudler 1.19x |

### CTC — CPU

| Clip | Lang | Duration s | qvac RTF | mudler RTF | Faster |
|------|------|-----------:|---------:|-----------:|--------|
| jfk | en | 11.0 | 0.1253 | 0.1516 | qvac 1.21x |
| alice | en | 20.1 | 0.1128 | 0.1468 | qvac 1.30x |
| croatian | hr | 27.4 | 0.1101 | 0.1468 | qvac 1.33x |
| french | fr | 29.4 | 0.1147 | 0.1489 | qvac 1.30x |
| spanish60 | es | 60.0 | 0.1458 | 0.1773 | qvac 1.22x |

### CTC — Metal

| Clip | Lang | Duration s | qvac RTF | mudler RTF | Faster |
|------|------|-----------:|---------:|-----------:|--------|
| jfk | en | 11.0 | 0.0226 | 0.0217 | mudler 1.04x |
| alice | en | 20.1 | 0.0259 | 0.0264 | qvac 1.02x |
| croatian | hr | 27.4 | 0.0292 | 0.0315 | qvac 1.08x |
| french | fr | 29.4 | 0.0295 | 0.0327 | qvac 1.11x |
| spanish60 | es | 60.0 | 0.0384 | 0.0387 | qvac 1.01x |

### EOU — CPU

| Clip | Lang | Duration s | qvac RTF | mudler RTF | Faster |
|------|------|-----------:|---------:|-----------:|--------|
| jfk | en | 11.0 | 0.0568 | 0.0397 | mudler 1.43x |
| alice | en | 20.1 | 0.0504 | 0.0437 | mudler 1.15x |
| croatian | hr | 27.4 | 0.0419 | 0.0432 | qvac 1.03x |
| french | fr | 29.4 | 0.0424 | 0.0448 | qvac 1.05x |
| spanish60 | es | 60.0 | 0.0464 | 0.0551 | qvac 1.19x |

### EOU — Metal

| Clip | Lang | Duration s | qvac RTF | mudler RTF | Faster |
|------|------|-----------:|---------:|-----------:|--------|
| jfk | en | 11.0 | 0.0186 | 0.0161 | mudler 1.16x |
| alice | en | 20.1 | 0.0203 | 0.0191 | mudler 1.06x |
| croatian | hr | 27.4 | 0.0203 | 0.0189 | mudler 1.07x |
| french | fr | 29.4 | 0.0213 | 0.0192 | mudler 1.11x |
| spanish60 | es | 60.0 | 0.0275 | 0.0232 | mudler 1.19x |

## 3. Accuracy (WER)

Reference WER uses ground-truth transcripts (English clips). Agreement = WER between the two engines (proxy for divergence; 0% = byte-identical word stream). Non-English WER only shown for the multilingual TDT model.

| Model | Backend | Clip | Lang | qvac WER (ref) | mudler WER (ref) | Agreement |
|-------|---------|------|------|---------------:|-----------------:|----------:|
| TDT | CPU | jfk | en | 0.0% | 0.0% | 0.0% |
| TDT | CPU | alice | en | 0.0% | 0.0% | 0.0% |
| TDT | CPU | croatian | hr | n/a | n/a | 10.3% |
| TDT | CPU | french | fr | n/a | n/a | 0.0% |
| TDT | CPU | spanish60 | es | n/a | n/a | 56.4% |
| TDT | Metal | jfk | en | 0.0% | 0.0% | 0.0% |
| TDT | Metal | alice | en | 0.0% | 0.0% | 0.0% |
| TDT | Metal | croatian | hr | n/a | n/a | 22.5% |
| TDT | Metal | french | fr | n/a | n/a | 31.3% |
| TDT | Metal | spanish60 | es | n/a | n/a | 49.3% |
| CTC | CPU | jfk | en | 0.0% | 0.0% | 0.0% |
| CTC | CPU | alice | en | 0.0% | 0.0% | 0.0% |
| CTC | Metal | jfk | en | 0.0% | 0.0% | 0.0% |
| CTC | Metal | alice | en | 0.0% | 0.0% | 0.0% |
| EOU | CPU | jfk | en | 0.0% | 4.5% | 4.5% |
| EOU | CPU | alice | en | 0.0% | 1.8% | 1.8% |
| EOU | Metal | jfk | en | 0.0% | 4.5% | 4.5% |
| EOU | Metal | alice | en | 0.0% | 1.8% | 1.8% |

## 4. Multilingual accuracy — FLEURS ground truth (TDT, Metal)

Real WER against FLEURS reference transcripts (corpus-level: total word edits / total reference words). TDT 0.6B v3 is the only multilingual model; CTC/EOU are English-only and excluded here.

| Language | Utterances | Ref words | qvac WER | mudler WER | Closer to reference |
|----------|-----------:|----------:|---------:|-----------:|---------------------|
| Bulgarian (bg) | 12 | 264 | 12.9% | 10.2% | mudler |
| Croatian (hr) | 12 | 204 | 22.5% | 23.0% | qvac |
| Czech (cs) | 12 | 230 | 3.0% | 3.0% | qvac |
| Danish (da) | 12 | 262 | 15.6% | 11.1% | mudler |
| Dutch (nl) | 12 | 210 | 5.2% | 5.7% | qvac |
| English (en) | 12 | 278 | 12.6% | 5.4% | mudler |
| Estonian (et) | 12 | 179 | 17.3% | 17.3% | qvac |
| Finnish (fi) | 12 | 147 | 5.4% | 4.1% | mudler |
| French (fr) | 12 | 265 | 12.8% | 6.8% | mudler |
| German (de) | 12 | 207 | 10.6% | 3.9% | mudler |
| Greek (el) | 12 | 277 | 48.7% | 36.1% | mudler |
| Hungarian (hu) | 12 | 247 | 15.8% | 14.2% | mudler |
| Italian (it) | 12 | 279 | 4.3% | 3.6% | mudler |
| Latvian (lv) | 12 | 216 | 15.3% | 12.0% | mudler |
| Lithuanian (lt) | 12 | 206 | 19.9% | 18.0% | mudler |
| Maltese (mt) | 12 | 226 | 24.3% | 20.4% | mudler |
| Polish (pl) | 12 | 163 | 6.7% | 5.5% | mudler |
| Portuguese (pt) | 12 | 298 | 7.0% | 2.7% | mudler |
| Romanian (ro) | 12 | 299 | 9.7% | 8.4% | mudler |
| Russian (ru) | 12 | 197 | 3.0% | 2.5% | mudler |
| Slovak (sk) | 12 | 168 | 4.2% | 2.4% | mudler |
| Slovenian (sl) | 12 | 281 | 16.0% | 14.2% | mudler |
| Spanish (es) | 12 | 306 | 1.6% | 1.0% | mudler |
| Swedish (sv) | 12 | 281 | 23.8% | 24.2% | qvac |
| Ukrainian (uk) | 12 | 195 | 8.2% | 7.2% | mudler |

## Feature differences

Both are ggml ports of the same NVIDIA Parakeet checkpoints, but they target different products, so their feature sets diverge.

### Only in qvac (`transcription-parakeet`)
- **Speaker diarization** — Sortformer v1 / v2 / v2.1 with NeMo Audio-Online Speaker Cache (AOSC) so speakers rebind to their slot across gaps. mudler has no diarization at all.
- **Speaker-attributed transcription** ("who said what") — ASR + Sortformer combined into one tagged transcript.
- **Live duplex streaming + microphone** — Mode 3 cache-aware chunks (left-context / right-lookahead), `<EOU>` turn boundaries, `StreamEvent` callbacks, energy VAD, and `live-mic` / `live-mic-attributed` example apps.
- **Mobile & embedded reach** — iOS and Android (arm64) builds, plus the **OpenCL** backend for Adreno GPUs.
- **Runtime integration** — ships as a Bare/Node native addon driven from the QVAC SDK (JS API, P2P, batched `run()` / streaming `runStreaming()`), not just a CLI.

### Only in mudler (`parakeet.cpp`)
- **CUDA (NVIDIA) and HIP/ROCm (AMD) backends** — qvac is Metal / Vulkan / OpenCL only (no CUDA).
- **K-quants** (`q4_k`, `q5_k`, `q6_k`) via `parakeet-cli quantize`. qvac ships `f16 / q8_0 / q5_0 / q4_0` only.
- **More & larger checkpoints** — 1.1B family (CTC / RNNT / TDT / hybrid TDT+CTC), 110M hybrid, RNNT 0.6B, and **nemotron-3.5 streaming multilingual** (40+ locales, prompt-conditioned, `--lang`).
- **Batched decode** (`bench-batch`, `--batch-sizes`) and a `bench-decode` microbenchmark.
- **Distribution surface** — flat C-API (`parakeet_capi.h`) + shared lib for dlopen/FFI/LocalAI, prebuilt CLI binaries for 5 platforms, and Docker images (CPU + CUDA, multi-arch) on GHCR.
- **Word/segment timestamps** (`--timestamps`).

### Shared by both
CTC + TDT + EOU transcription · `q8_0` / `f16` · CPU + Metal + Vulkan · ggml-based · log-mel front-end on GPU · WER-0 parity vs NeMo on clean English.

## Benchmark caveats
- CTC and EOU are English-only; their transcripts on non-English clips are expected to be wrong (timing still valid).
- GGUF schemas are not interchangeable (verified both directions): qvac uses renamed `blk`-style tensors + `parakeet.*` KV; mudler keeps verbatim NeMo names. Each engine runs its own native q8_0 file.
- On NVIDIA the GPU column means **CUDA for mudler / Vulkan for qvac** unless mudler is also built with Vulkan; qvac's actual backend is recorded as `backendId` in the JSON.
