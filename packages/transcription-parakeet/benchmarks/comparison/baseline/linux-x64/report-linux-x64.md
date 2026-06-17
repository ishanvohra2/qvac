# Parakeet Engine Comparison: qvac vs mudler/parakeet.cpp

Generated: 2026-06-17T10:14:59.387Z  
Platform: `linux-x64` (NVIDIA RTX 4000 Ada, Vulkan)  
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
| TDT | CPU | qvac | 1265.9 | 0.0629 |  |
| TDT | CPU | mudler | 773.7 | 0.0384 | **mudler** 1.64x |
| TDT | Vulkan | qvac | 43.7 | 0.0022 | **qvac** 7.57x |
| TDT | Vulkan | mudler | 331.0 | 0.0164 |  |
| CTC | CPU | qvac | 1142.4 | 0.0567 |  |
| CTC | CPU | mudler | 715.8 | 0.0356 | **mudler** 1.60x |
| CTC | Vulkan | qvac | 21.2 | 0.0011 | **qvac** 1.37x |
| CTC | Vulkan | mudler | 29.1 | 0.0014 |  |
| EOU | CPU | qvac | 482.9 | 0.0240 |  |
| EOU | CPU | mudler | 329.6 | 0.0164 | **mudler** 1.46x |
| EOU | Vulkan | qvac | 57.0 | 0.0028 |  |
| EOU | Vulkan | mudler | 50.3 | 0.0025 | **mudler** 1.13x |

## 2. RTF vs clip duration (speed stability across lengths)

### TDT — CPU

| Clip | Lang | Duration s | qvac RTF | mudler RTF | Faster |
|------|------|-----------:|---------:|-----------:|--------|
| jfk | en | 11.0 | 0.0612 | 0.0363 | mudler 1.69x |
| alice | en | 20.1 | 0.0629 | 0.0384 | mudler 1.64x |
| croatian | hr | 27.4 | 0.0638 | 0.0392 | mudler 1.63x |
| french | fr | 29.4 | 0.0647 | 0.0382 | mudler 1.69x |
| spanish60 | es | 60.0 | 0.0712 | 0.0459 | mudler 1.55x |

### TDT — Vulkan

| Clip | Lang | Duration s | qvac RTF | mudler RTF | Faster |
|------|------|-----------:|---------:|-----------:|--------|
| jfk | en | 11.0 | 0.0044 | 0.0035 | mudler 1.25x |
| alice | en | 20.1 | 0.0022 | 0.0164 | qvac 7.57x |
| croatian | hr | 27.4 | 0.0020 | 0.0024 | qvac 1.20x |
| french | fr | 29.4 | 0.0022 | 0.0028 | qvac 1.23x |
| spanish60 | es | 60.0 | 0.0027 | 0.0091 | qvac 3.33x |

### CTC — CPU

| Clip | Lang | Duration s | qvac RTF | mudler RTF | Faster |
|------|------|-----------:|---------:|-----------:|--------|
| jfk | en | 11.0 | 0.0556 | 0.0337 | mudler 1.65x |
| alice | en | 20.1 | 0.0567 | 0.0356 | mudler 1.60x |
| croatian | hr | 27.4 | 0.0580 | 0.0367 | mudler 1.58x |
| french | fr | 29.4 | 0.0580 | 0.0353 | mudler 1.64x |
| spanish60 | es | 60.0 | 0.0648 | 0.0435 | mudler 1.49x |

### CTC — Vulkan

| Clip | Lang | Duration s | qvac RTF | mudler RTF | Faster |
|------|------|-----------:|---------:|-----------:|--------|
| jfk | en | 11.0 | 0.0014 | 0.0020 | qvac 1.42x |
| alice | en | 20.1 | 0.0011 | 0.0014 | qvac 1.37x |
| croatian | hr | 27.4 | 0.0010 | 0.0013 | qvac 1.40x |
| french | fr | 29.4 | 0.0009 | 0.0014 | qvac 1.45x |
| spanish60 | es | 60.0 | 0.0009 | 0.0016 | qvac 1.74x |

### EOU — CPU

| Clip | Lang | Duration s | qvac RTF | mudler RTF | Faster |
|------|------|-----------:|---------:|-----------:|--------|
| jfk | en | 11.0 | 0.0228 | 0.0143 | mudler 1.59x |
| alice | en | 20.1 | 0.0240 | 0.0164 | mudler 1.46x |
| croatian | hr | 27.4 | 0.0239 | 0.0165 | mudler 1.45x |
| french | fr | 29.4 | 0.0242 | 0.0169 | mudler 1.43x |
| spanish60 | es | 60.0 | 0.0281 | 0.0206 | mudler 1.36x |

### EOU — Vulkan

| Clip | Lang | Duration s | qvac RTF | mudler RTF | Faster |
|------|------|-----------:|---------:|-----------:|--------|
| jfk | en | 11.0 | 0.0026 | 0.0023 | mudler 1.16x |
| alice | en | 20.1 | 0.0028 | 0.0025 | mudler 1.13x |
| croatian | hr | 27.4 | 0.0021 | 0.0019 | mudler 1.11x |
| french | fr | 29.4 | 0.0018 | 0.0018 | mudler 1.02x |
| spanish60 | es | 60.0 | 0.0022 | 0.0023 | qvac 1.03x |

## 3. Accuracy (WER)

Reference WER uses ground-truth transcripts (English clips). Agreement = WER between the two engines (proxy for divergence; 0% = byte-identical word stream). Non-English WER only shown for the multilingual TDT model.

| Model | Backend | Clip | Lang | qvac WER (ref) | mudler WER (ref) | Agreement |
|-------|---------|------|------|---------------:|-----------------:|----------:|
| TDT | CPU | jfk | en | 0.0% | 0.0% | 0.0% |
| TDT | CPU | alice | en | 0.0% | 0.0% | 0.0% |
| TDT | CPU | croatian | hr | n/a | n/a | 10.3% |
| TDT | CPU | french | fr | n/a | n/a | 0.0% |
| TDT | CPU | spanish60 | es | n/a | n/a | 47.2% |
| TDT | Vulkan | jfk | en | 0.0% | 0.0% | 0.0% |
| TDT | Vulkan | alice | en | 0.0% | 0.0% | 0.0% |
| TDT | Vulkan | croatian | hr | n/a | n/a | 20.0% |
| TDT | Vulkan | french | fr | n/a | n/a | 34.4% |
| TDT | Vulkan | spanish60 | es | n/a | n/a | 46.6% |
| CTC | CPU | jfk | en | 0.0% | 0.0% | 0.0% |
| CTC | CPU | alice | en | 0.0% | 0.0% | 0.0% |
| CTC | Vulkan | jfk | en | 0.0% | 0.0% | 0.0% |
| CTC | Vulkan | alice | en | 0.0% | 0.0% | 0.0% |
| EOU | CPU | jfk | en | 0.0% | 4.5% | 4.5% |
| EOU | CPU | alice | en | 0.0% | 1.8% | 1.8% |
| EOU | Vulkan | jfk | en | 0.0% | 4.5% | 4.5% |
| EOU | Vulkan | alice | en | 0.0% | 1.8% | 1.8% |

## 4. Multilingual accuracy — FLEURS ground truth (TDT, Vulkan)

Real WER against FLEURS reference transcripts (corpus-level: total word edits / total reference words). TDT 0.6B v3 is the only multilingual model; CTC/EOU are English-only and excluded here.

| Language | Utterances | Ref words | qvac WER | mudler WER | Closer to reference |
|----------|-----------:|----------:|---------:|-----------:|---------------------|
| French (fr) | 12 | 265 | 12.8% | 7.2% | mudler |
| Spanish (es) | 12 | 306 | 1.6% | 1.0% | mudler |
| Croatian (hr) | 12 | 204 | 21.6% | 21.6% | qvac |

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
