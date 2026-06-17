# qvac `transcription-parakeet` vs `mudler/parakeet.cpp` — Feature Comparison

A platform-independent comparison of the two ggml ports of NVIDIA's Parakeet
(FastConformer) speech-recognition family. Both run pure C++ inference on
`ggml` with no Python/PyTorch at runtime, and both originate from the same
NVIDIA checkpoints — but they target different products, so their feature sets
diverge.

- **qvac** — [`tetherto/qvac-ext-lib-whisper.cpp` → `parakeet-cpp`](https://github.com/tetherto/qvac-ext-lib-whisper.cpp), consumed by the `transcription-parakeet` Bare/Node addon in the QVAC SDK.
- **mudler** — [`mudler/parakeet.cpp`](https://github.com/mudler/parakeet.cpp), a standalone CLI / shared library (the engine used by LocalAI).

> This document covers only facts that do **not** depend on benchmark runs
> (capabilities, platforms, models, formats). For measured speed/accuracy
> numbers see the per-platform reports under `baseline/<platform>/`.

---

## Model types (ASR decoders)

All Parakeet variants share the same FastConformer audio encoder; they differ in
how they turn encoder output into text. Three decoders are commonly compared:

| Model | Full name | How it works | Trade-off | Languages |
|-------|-----------|--------------|-----------|-----------|
| **CTC** | Connectionist Temporal Classification | Non-autoregressive: predicts one token per audio frame (plus a "blank"), then collapses repeats/blanks into text in a single pass. | Fastest & simplest; no explicit duration model, slightly weaker on hard audio. | English |
| **TDT** | Token-and-Duration Transducer (RNN-T family) | A transducer that predicts each token *and how many frames to skip* (its duration), striding over audio instead of stepping frame-by-frame. | Best accuracy + punctuation/capitalization (PnC); multilingual. Slightly heavier decoder. | ~25 (v3) |
| **EOU** | End-of-Utterance streaming (RNN-T + `<EOU>`) | A small 120M streaming model that also emits an `<EOU>` token to detect when a speaker finished their turn. | Built for low-latency live conversation / turn-taking, not peak accuracy. | English |

Other decoders that exist in the ecosystem: **RNNT** (plain transducer),
**hybrid TDT+CTC** (one checkpoint, both heads), and **Sortformer** (speaker
*diarization* — "who spoke when", not transcription).

---

## Platform & GPU support matrix

What each project supports out of the box. CPU is available everywhere;
**bold** = GPU acceleration.

| Platform / Arch | qvac `transcription-parakeet` | `mudler/parakeet.cpp` |
|-----------------|-------------------------------|-----------------------|
| macOS arm64 | CPU + **Metal** | CPU + **Metal** |
| macOS x64 | CPU + **Metal** | CPU only |
| iOS arm64 | CPU + **Metal** | — not supported |
| Linux x64 | CPU + **Vulkan** | CPU + **Vulkan** + **CUDA** |
| Linux arm64 | CPU + **Vulkan** | CPU only |
| Android arm64 | CPU + **Vulkan / OpenCL** | — not supported |
| Windows x64 | CPU + **Vulkan** | CPU + **Vulkan** + **CUDA** |
| AMD (ROCm/HIP) | — | source build (`PARAKEET_GGML_HIP`) |

**GPU backends:** Metal (both) · Vulkan (both) · **OpenCL → qvac only**
(Android/Adreno) · **CUDA + HIP → mudler only** (NVIDIA / AMD).

Notes:
- qvac selects exactly one GPU backend at vcpkg/compile time and falls back to
  CPU at runtime if it fails to initialise; it reflects the active backend via
  `backend_device` / `backend_name`.
- mudler auto-selects the first GPU device the ggml registry reports (including
  integrated GPUs) and can be overridden with `PARAKEET_DEVICE` (e.g. `cpu`,
  `CUDA0`, `Vulkan1`).
- mudler ships prebuilt CLI binaries for 5 platforms (Linux x64 cpu/vulkan/cuda,
  Linux arm64 cpu, macOS arm64 metal, macOS x64 cpu, Windows x64 cpu/vulkan/cuda)
  plus Docker images. qvac ships as a native addon prebuild per QVAC platform.

---

## Quantization formats

| Format | qvac | mudler |
|--------|:----:|:------:|
| `f32` | ✅ | ✅ (default) |
| `f16` | ✅ | ✅ |
| `q8_0` | ✅ | ✅ |
| `q5_0` | ✅ | ✅ (CLI quantize) |
| `q4_0` | ✅ | ✅ (CLI quantize) |
| `q4_k` / `q5_k` / `q6_k` (K-quants) | ❌ | ✅ (CLI quantize) |

Both Python converters emit up to `q8_0`; mudler additionally re-quantizes an
F32/F16 GGUF to K-quants via `parakeet-cli quantize`.

> **GGUF schemas are not interchangeable** between the two (verified both
> directions): qvac uses renamed `blk`-style tensors + `parakeet.*` KV metadata,
> while mudler keeps verbatim NeMo tensor names. Each engine must load its own
> native GGUF even at the same quant level.

---

## Checkpoint coverage

| Checkpoint | Type | qvac | mudler |
|------------|------|:----:|:------:|
| `parakeet-ctc-0.6b` | CTC, EN | ✅ | ✅ |
| `parakeet-ctc-1.1b` | CTC, EN | ✅ | ✅ |
| `parakeet-tdt-0.6b-v3` | TDT, ~25 langs | ✅ | ✅ |
| `parakeet-tdt-0.6b-v2` | TDT, EN | — | ✅ |
| `parakeet-tdt-1.1b` | TDT, EN | ✅ | ✅ |
| `parakeet-rnnt-0.6b` / `-1.1b` | RNNT, EN | — | ✅ |
| `parakeet-tdt_ctc-110m` / `-1.1b` | hybrid TDT+CTC | — | ✅ |
| `parakeet_realtime_eou_120m-v1` | EOU streaming, EN | ✅ | ✅ |
| `nemotron-3.5-asr-streaming-0.6b` | RNNT streaming, 40+ locales | — | ✅ |
| `diar_sortformer_4spk-v1` | Sortformer diarization | ✅ | — |
| `diar_streaming_sortformer_4spk-v2 / v2.1` | Sortformer streaming (AOSC) | ✅ | — |

---

## Feature differences

### Only in qvac (`transcription-parakeet`)
- **Speaker diarization** — Sortformer v1 / v2 / v2.1 with NeMo Audio-Online
  Speaker Cache (AOSC), so speakers rebind to their original slot across long
  gaps. mudler has no diarization at all.
- **Speaker-attributed transcription** ("who said what") — ASR + Sortformer
  combined into a single tagged transcript (`transcribe_with_speakers`,
  `live-mic-attributed`).
- **Live duplex streaming + microphone** — Mode 3 cache-aware chunks
  (left-context / right-lookahead), `<EOU>` turn boundaries, `StreamEvent`
  callbacks, energy VAD, and `live-mic` / `live-mic-attributed` example apps.
- **Mobile & embedded reach** — iOS and Android (arm64) builds, plus the
  **OpenCL** backend for Adreno GPUs.
- **Runtime integration** — ships as a Bare/Node native addon driven from the
  QVAC SDK (JS API, P2P, batched `run()` / streaming `runStreaming()`), with a
  shared `qvac-speech-` ggml flavour that coexists with other QVAC speech models
  on one device.

### Only in mudler (`parakeet.cpp`)
- **CUDA (NVIDIA) and HIP/ROCm (AMD) backends** — qvac is Metal / Vulkan /
  OpenCL only (no CUDA).
- **K-quants** (`q4_k`, `q5_k`, `q6_k`) via `parakeet-cli quantize`.
- **More & larger checkpoints** — 1.1B family (CTC / RNNT / TDT / hybrid
  TDT+CTC), 110M hybrid, RNNT 0.6B, and **nemotron-3.5 streaming multilingual**
  (40+ locales, prompt-conditioned, `--lang`).
- **Batched decode** (`bench-batch`, `--batch-sizes`) and a `bench-decode`
  microbenchmark.
- **Distribution surface** — flat C-API (`parakeet_capi.h`) + shared lib for
  dlopen / FFI / LocalAI, prebuilt CLI binaries for 5 platforms, and Docker
  images (CPU + CUDA, multi-arch) on GHCR.
- **Word/segment timestamps** (`--timestamps`).

### Shared by both
CTC + TDT + EOU transcription · `q8_0` / `f16` · CPU + Metal + Vulkan ·
ggml-based · log-mel front-end on GPU · WER-0 parity vs NeMo on clean English.

---

## Summary

- **qvac** is the broader *product* engine: diarization, speaker attribution,
  live streaming, and mobile (iOS/Android/OpenCL) reach, wired into the QVAC SDK.
- **mudler** is the broader *standalone* engine: more checkpoints (incl. 1.1B,
  hybrid, RNNT, nemotron multilingual), K-quants, CUDA/HIP, batching, and a
  C-API/Docker distribution surface, but no diarization and no mobile.

---

## Acronyms

- **CTC** (Connectionist Temporal Classification) — a non-autoregressive decoder that predicts one token per audio frame plus a "blank", then collapses repeats and blanks into the final text in a single fast pass.
- **TDT** (Token-and-Duration Transducer) — an RNN-T-family decoder that predicts each token *and* how many audio frames to skip next, letting it stride efficiently over the audio while producing punctuated, multilingual output.
- **EOU** (End-of-Utterance) — a small streaming RNN-T model that, alongside transcription, emits an `<EOU>` token to detect when a speaker has finished their turn, enabling low-latency live conversation.
