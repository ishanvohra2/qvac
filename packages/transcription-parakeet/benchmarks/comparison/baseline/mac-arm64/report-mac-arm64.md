# Parakeet Engine Comparison: qvac vs mudler/parakeet.cpp

Generated: 2026-06-17T08:14:54.856Z  
Platform: `darwin-arm64` (Apple Silicon, Metal)  
Quant: `q8_0` · Threads: 4 · Warmup: 1 · Timed reps: 5

**RTF** = proc/audio (lower is faster) · **WER** lower is better.

> qvac time = full JS `run()` wall (product-level, includes Bare/JS tax); mudler time = engine-only `transcribe_pcm`. Same canonical clips, same threads, same quant level. Each engine loads its own native q8_0 GGUF (the two schemas are not interchangeable).

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
| TDT | CPU | qvac | 1547.8 | 0.0769 | **qvac** 1.37x |
| TDT | CPU | mudler | 2121.2 | 0.1054 |  |
| TDT | Metal | qvac | 521.3 | 0.0259 |  |
| TDT | Metal | mudler | 515.5 | 0.0256 | **mudler** 1.01x |
| CTC | CPU | qvac | 1546.2 | 0.0768 | **qvac** 1.41x |
| CTC | CPU | mudler | 2181.9 | 0.1084 |  |
| CTC | Metal | qvac | 450.0 | 0.0224 |  |
| CTC | Metal | mudler | 427.2 | 0.0212 | **mudler** 1.05x |
| EOU | CPU | qvac | 721.2 | 0.0358 |  |
| EOU | CPU | mudler | 696.6 | 0.0346 | **mudler** 1.04x |
| EOU | Metal | qvac | 397.3 | 0.0197 |  |
| EOU | Metal | mudler | 347.0 | 0.0172 | **mudler** 1.14x |

## 2. RTF vs clip duration (speed stability across lengths)

### TDT — CPU

| Clip | Lang | Duration s | qvac RTF | mudler RTF | Faster |
|------|------|-----------:|---------:|-----------:|--------|
| jfk | en | 11.0 | 0.0751 | 0.0986 | qvac 1.31x |
| alice | en | 20.1 | 0.0769 | 0.1054 | qvac 1.37x |
| croatian | hr | 27.4 | 0.0841 | 0.1048 | qvac 1.25x |
| french | fr | 29.4 | 0.0872 | 0.1169 | qvac 1.34x |
| spanish60 | es | 60.0 | 0.1008 | 0.1273 | qvac 1.26x |

### TDT — Metal

| Clip | Lang | Duration s | qvac RTF | mudler RTF | Faster |
|------|------|-----------:|---------:|-----------:|--------|
| jfk | en | 11.0 | 0.0250 | 0.0235 | mudler 1.06x |
| alice | en | 20.1 | 0.0259 | 0.0256 | mudler 1.01x |
| croatian | hr | 27.4 | 0.0273 | 0.0255 | mudler 1.07x |
| french | fr | 29.4 | 0.0283 | 0.0273 | mudler 1.04x |
| spanish60 | es | 60.0 | 0.0340 | 0.0316 | mudler 1.08x |

### CTC — CPU

| Clip | Lang | Duration s | qvac RTF | mudler RTF | Faster |
|------|------|-----------:|---------:|-----------:|--------|
| jfk | en | 11.0 | 0.0824 | 0.0979 | qvac 1.19x |
| alice | en | 20.1 | 0.0768 | 0.1084 | qvac 1.41x |
| croatian | hr | 27.4 | 0.0831 | 0.1044 | qvac 1.26x |
| french | fr | 29.4 | 0.0830 | 0.1121 | qvac 1.35x |
| spanish60 | es | 60.0 | 0.1016 | 0.1293 | qvac 1.27x |

### CTC — Metal

| Clip | Lang | Duration s | qvac RTF | mudler RTF | Faster |
|------|------|-----------:|---------:|-----------:|--------|
| jfk | en | 11.0 | 0.0222 | 0.0208 | mudler 1.07x |
| alice | en | 20.1 | 0.0224 | 0.0212 | mudler 1.05x |
| croatian | hr | 27.4 | 0.0236 | 0.0229 | mudler 1.03x |
| french | fr | 29.4 | 0.0239 | 0.0230 | mudler 1.04x |
| spanish60 | es | 60.0 | 0.0306 | 0.0279 | mudler 1.09x |

### EOU — CPU

| Clip | Lang | Duration s | qvac RTF | mudler RTF | Faster |
|------|------|-----------:|---------:|-----------:|--------|
| jfk | en | 11.0 | 0.0366 | 0.0306 | mudler 1.20x |
| alice | en | 20.1 | 0.0358 | 0.0346 | mudler 1.04x |
| croatian | hr | 27.4 | 0.0370 | 0.0343 | mudler 1.08x |
| french | fr | 29.4 | 0.0370 | 0.0351 | mudler 1.06x |
| spanish60 | es | 60.0 | 0.0404 | 0.0409 | qvac 1.01x |

### EOU — Metal

| Clip | Lang | Duration s | qvac RTF | mudler RTF | Faster |
|------|------|-----------:|---------:|-----------:|--------|
| jfk | en | 11.0 | 0.0183 | 0.0151 | mudler 1.22x |
| alice | en | 20.1 | 0.0197 | 0.0172 | mudler 1.14x |
| croatian | hr | 27.4 | 0.0191 | 0.0156 | mudler 1.22x |
| french | fr | 29.4 | 0.0196 | 0.0157 | mudler 1.25x |
| spanish60 | es | 60.0 | 0.0225 | 0.0202 | mudler 1.11x |

## 3. Accuracy (WER)

Reference WER uses ground-truth transcripts (English clips). Agreement = WER between the two engines (proxy for divergence; 0% = byte-identical word stream). Non-English WER only shown for the multilingual TDT model.

| Model | Backend | Clip | Lang | qvac WER (ref) | mudler WER (ref) | Agreement |
|-------|---------|------|------|---------------:|-----------------:|----------:|
| TDT | CPU | jfk | en | 0.0% | 0.0% | 0.0% |
| TDT | CPU | alice | en | 0.0% | 0.0% | 0.0% |
| TDT | CPU | croatian | hr | n/a | n/a | 9.8% |
| TDT | CPU | french | fr | n/a | n/a | 1.4% |
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
