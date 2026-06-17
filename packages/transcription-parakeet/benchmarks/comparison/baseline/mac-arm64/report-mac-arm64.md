# Parakeet Engine Comparison: qvac vs mudler/parakeet.cpp

Generated: 2026-06-17T12:50:04.605Z  
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
| TDT | CPU | qvac | 1719.0 | 0.0854 | **qvac** 1.24x |
| TDT | CPU | mudler | 2132.5 | 0.1059 |  |
| TDT | Metal | qvac | 626.9 | 0.0311 | **qvac** 1.12x |
| TDT | Metal | mudler | 701.7 | 0.0348 |  |
| CTC | CPU | qvac | 2469.4 | 0.1227 | **qvac** 1.18x |
| CTC | CPU | mudler | 2903.5 | 0.1442 |  |
| CTC | Metal | qvac | 570.4 | 0.0283 |  |
| CTC | Metal | mudler | 496.8 | 0.0247 | **mudler** 1.15x |
| EOU | CPU | qvac | 928.8 | 0.0461 |  |
| EOU | CPU | mudler | 859.9 | 0.0427 | **mudler** 1.08x |
| EOU | Metal | qvac | 408.8 | 0.0203 |  |
| EOU | Metal | mudler | 393.4 | 0.0195 | **mudler** 1.04x |

## 2. RTF vs clip duration (speed stability across lengths)

### TDT — CPU

| Clip | Lang | Duration s | qvac RTF | mudler RTF | Faster |
|------|------|-----------:|---------:|-----------:|--------|
| jfk | en | 11.0 | 0.0843 | 0.1021 | qvac 1.21x |
| alice | en | 20.1 | 0.0854 | 0.1059 | qvac 1.24x |
| croatian | hr | 27.4 | 0.0832 | 0.1065 | qvac 1.28x |
| french | fr | 29.4 | 0.0915 | 0.1201 | qvac 1.31x |
| spanish60 | es | 60.0 | 0.1012 | 0.1564 | qvac 1.55x |

### TDT — Metal

| Clip | Lang | Duration s | qvac RTF | mudler RTF | Faster |
|------|------|-----------:|---------:|-----------:|--------|
| jfk | en | 11.0 | 0.0275 | 0.0266 | mudler 1.03x |
| alice | en | 20.1 | 0.0311 | 0.0348 | qvac 1.12x |
| croatian | hr | 27.4 | 0.0355 | 0.0384 | qvac 1.08x |
| french | fr | 29.4 | 0.0399 | 0.0380 | mudler 1.05x |
| spanish60 | es | 60.0 | 0.0483 | 0.0448 | mudler 1.08x |

### CTC — CPU

| Clip | Lang | Duration s | qvac RTF | mudler RTF | Faster |
|------|------|-----------:|---------:|-----------:|--------|
| jfk | en | 11.0 | 0.1475 | 0.1411 | mudler 1.04x |
| alice | en | 20.1 | 0.1227 | 0.1442 | qvac 1.18x |
| croatian | hr | 27.4 | 0.1225 | 0.1524 | qvac 1.24x |
| french | fr | 29.4 | 0.1257 | 0.1690 | qvac 1.34x |
| spanish60 | es | 60.0 | 0.1427 | 0.1880 | qvac 1.32x |

### CTC — Metal

| Clip | Lang | Duration s | qvac RTF | mudler RTF | Faster |
|------|------|-----------:|---------:|-----------:|--------|
| jfk | en | 11.0 | 0.0220 | 0.0199 | mudler 1.10x |
| alice | en | 20.1 | 0.0283 | 0.0247 | mudler 1.15x |
| croatian | hr | 27.4 | 0.0332 | 0.0275 | mudler 1.21x |
| french | fr | 29.4 | 0.0307 | 0.0282 | mudler 1.09x |
| spanish60 | es | 60.0 | 0.0396 | 0.0331 | mudler 1.20x |

### EOU — CPU

| Clip | Lang | Duration s | qvac RTF | mudler RTF | Faster |
|------|------|-----------:|---------:|-----------:|--------|
| jfk | en | 11.0 | 0.0471 | 0.0387 | mudler 1.22x |
| alice | en | 20.1 | 0.0461 | 0.0427 | mudler 1.08x |
| croatian | hr | 27.4 | 0.0419 | 0.0449 | qvac 1.07x |
| french | fr | 29.4 | 0.0459 | 0.0483 | qvac 1.05x |
| spanish60 | es | 60.0 | 0.0488 | 0.0568 | qvac 1.16x |

### EOU — Metal

| Clip | Lang | Duration s | qvac RTF | mudler RTF | Faster |
|------|------|-----------:|---------:|-----------:|--------|
| jfk | en | 11.0 | 0.0189 | 0.0168 | mudler 1.12x |
| alice | en | 20.1 | 0.0203 | 0.0195 | mudler 1.04x |
| croatian | hr | 27.4 | 0.0200 | 0.0193 | mudler 1.04x |
| french | fr | 29.4 | 0.0209 | 0.0201 | mudler 1.04x |
| spanish60 | es | 60.0 | 0.0268 | 0.0242 | mudler 1.11x |

## 3. Accuracy (WER)

Reference WER uses ground-truth transcripts (English clips). Agreement = WER between the two engines (proxy for divergence; 0% = byte-identical word stream). Non-English WER only shown for the multilingual TDT model.

| Model | Backend | Clip | Lang | qvac WER (ref) | mudler WER (ref) | Agreement |
|-------|---------|------|------|---------------:|-----------------:|----------:|
| TDT | CPU | jfk | en | 0.0% | 0.0% | 0.0% |
| TDT | CPU | alice | en | 0.0% | 0.0% | 0.0% |
| TDT | CPU | croatian | hr | n/a | n/a | 9.8% |
| TDT | CPU | french | fr | n/a | n/a | 0.0% |
| TDT | CPU | spanish60 | es | n/a | n/a | 54.5% |
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
| French (fr) | 12 | 265 | 12.8% | 6.8% | mudler |
| Spanish (es) | 12 | 306 | 1.6% | 1.0% | mudler |
| Croatian (hr) | 12 | 204 | 22.5% | 23.0% | qvac |

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
