# qvac `parakeet-cpp` vs `mudler/parakeet.cpp` — Architecture & Feature Comparison

Both projects are ggml ports of NVIDIA's Parakeet (FastConformer) ASR family —
pure C++ inference, no Python/PyTorch at runtime, built from the same NVIDIA
checkpoints. But they were written for different products, so they decompose the
problem differently and expose very different APIs.

- **qvac** — [`tetherto/qvac-ext-lib-whisper.cpp` → `parakeet-cpp`](https://github.com/tetherto/qvac-ext-lib-whisper.cpp), the C++ engine wrapped by the `transcription-parakeet` Bare/Node addon in the QVAC SDK.
- **mudler** — [`mudler/parakeet.cpp`](https://github.com/mudler/parakeet.cpp), a standalone CLI / shared library (the engine used by LocalAI).

> This document covers facts that do **not** depend on benchmark runs
> (architecture, capabilities, platforms, models, formats). For measured
> speed/accuracy numbers see the per-platform reports under `baseline/<platform>/`.

---

## Architecture at a glance

| Axis | mudler `parakeet.cpp` | qvac `parakeet-cpp` |
|------|-----------------------|---------------------|
| Decomposition | By **NN component** (encoder, subsampling, attention, joint, prediction, search, …) | By **model family** behind one stateful `Engine` (ctc / tdt / eou / sortformer) |
| Public API | Flat **C-API** + minimal C++ free functions; decoder routed by `parakeet.arch` | Stateful **`Engine` class** (pimpl), load-once, rich methods + `StreamSession` objects |
| Graph build | Generic **`graph_builder` / `ggml_graph`** abstraction (`GraphInputPool`) | Per-model graph builders; encoder graph **shared** across CTC/TDT/EOU via metadata |
| Mel front-end | Split `mel.cpp` + `mel_gpu.cpp` + `fft.cpp` | Single `mel_preprocess.cpp` (GPU DFT-matmul) |
| Batching | `transducer_batch.cpp` (batched decode, `--batch-sizes`) | Single-stream only |
| Diarization | — none — | `parakeet_sortformer.cpp` + speaker attribution + energy VAD |
| ggml | Vendored **upstream** ggml submodule (`third_party/ggml`) | Forked **`qvac-ext-ggml@speech`** (shared speech-stack flavour + patches) |
| GPU backends | CUDA / HIP / Vulkan / Metal | Metal / Vulkan / OpenCL (no CUDA) |
| Ships as | Standalone CLI, shared lib, Docker | Subdir of a speech monorepo (shares one ggml with whisper / TTS), wrapped by an SDK addon |

---

## 1. Code organization

mudler decomposes the model **by neural-network layer** — each Conformer/transducer
piece is its own module, assembled through a generic graph builder:

```text
mudler/parakeet.cpp/src/   (~45 modules)
  encoder.cpp  subsampling.cpp  conformer.cpp  relpos_attention.cpp  pos_enc.cpp
  joint.cpp    prediction.cpp   rnnt.cpp       tdt.cpp               ctc_decoder.cpp
  search.cpp   transducer_batch.cpp            prompt_kernel.cpp
  graph_builder.hpp  ggml_graph.cpp            mel.cpp  mel_gpu.cpp  fft.cpp
  model.cpp    model_loader.cpp  tokenizer.cpp  parakeet_capi.cpp
```

qvac decomposes **by model family** — one engine, one file per decoder, with the
FastConformer encoder graph shared and selected from GGUF metadata:

```text
qvac parakeet-cpp/src/   (~12 modules)
  parakeet_engine.cpp        # one Engine, dispatches on metadata
  parakeet_ctc.cpp  parakeet_tdt.cpp  parakeet_eou.cpp  parakeet_sortformer.cpp
  mel_preprocess.cpp  sentencepiece_bpe.cpp  energy_vad.cpp
```

> Per qvac's README, *"EOU shares the same C++ graph path as CTC/TDT where
> weights allow"* — the encoder is one code path; only the decoder head differs.
> mudler instead builds one **fused** encoder graph through `GraphInputPool`,
> feeding host-computed inputs (mel transpose, masks, batch-norm fold) into the
> gallocr buffer after allocation.

---

## 2. API & usage model

**mudler** — a flat C-API for `dlopen`/FFI/LocalAI plus a thin C++ helper; the
decoder is chosen from the GGUF `parakeet.arch` value:

```c
// include/parakeet.h
int  parakeet_transcribe_file(const char* model_path, const char* wav_path, char** out);
void parakeet_free_string(char* s);
```

```cpp
// C++ helper: routes TDT vs CTC head by arch metadata (override optional)
enum class Decoder { kDefault, kCTC, kTDT };
std::string pk::transcribe(const std::string& model_path,
                           const std::string& wav_path,
                           Decoder decoder = Decoder::kDefault);
```

**qvac** — a stateful `Engine` (pimpl) that loads weights once and exposes
transcription, streaming, diarization, cancel, prewarm, and post-fallback backend
reflection. It's built to be wrapped by the Bare/Node addon:

```cpp
// include/parakeet/engine.h
parakeet::EngineOptions opts;
opts.model_gguf_path = "models/parakeet-tdt-0.6b-v3.q8_0.gguf";
opts.n_gpu_layers    = 1;       // 0 = CPU
opts.prewarm         = true;    // amortise first-call GPU pipeline compile

parakeet::Engine engine(opts);              // load once
auto r = engine.transcribe("clip.wav");     // mel + encoder + decode only
engine.stream_start(streamOpts, onSegment); // live duplex session
engine.diarize("clip.wav");                 // Sortformer "who spoke when"
engine.backend_name();                      // "Metal" / "Vulkan0" / "OpenCL" / "CPU"
```

So mudler is **function-oriented** (load-per-call helper + C-API), while qvac is
**object-oriented and stateful** (one `Engine`, session objects, a documented
threading/cancel model).

---

## 3. Model dispatch

Both read the decoder from GGUF metadata, but key differently:

- **mudler** routes on **`parakeet.arch`**: `tdt | hybrid_tdt_ctc | rnnt | hybrid_rnnt_ctc` → TDT head; `ctc` → CTC head.
- **qvac** routes on **`parakeet.model.type`**: `ctc | tdt | eou | sortformer`, selecting both the decoder and (for Sortformer) a different output path entirely.

GGUF tensor schemas are **not interchangeable** (verified both ways): qvac renames
tensors to `blk`-style names + `parakeet.*` KV metadata; mudler keeps verbatim
NeMo names. Each engine loads only its own native GGUF, even at the same quant.

---

## 4. Capability-shaping modules

These modules exist on one side and have no counterpart on the other:

- **mudler `transducer_batch.cpp`** — batched transducer decode (`bench-batch`, `--batch-sizes`); qvac is single-stream.
- **mudler `prompt_kernel.cpp`** — prompt-conditioning for the nemotron multilingual streaming model (`--lang`); qvac has no equivalent.
- **qvac `parakeet_sortformer.cpp` + `attributed.h`** — speaker diarization (incl. AOSC speaker-cache) and speaker-attributed transcription; mudler has none.
- **qvac `energy_vad.cpp` + `streaming.h`** — energy VAD + Mode 2/3 cache-aware streaming with `<EOU>` boundaries and `StreamEvent` callbacks.

---

## 5. ggml dependency & backend handling (the biggest infra difference)

**mudler** vendors **upstream ggml** as a submodule and lets ggml pick the device:

```bash
cmake -B build -DPARAKEET_GGML_CUDA=ON     # or _METAL / _VULKAN / _HIP
PARAKEET_DEVICE=CUDA1 ./parakeet ...        # override; auto-selects first GPU otherwise
```

**qvac** builds against a **forked `tetherto/qvac-ext-ggml` `speech` branch** — a
shared "ggml-speech" flavour (library prefix `qvac-speech-` / `speech-ggml-`) so
parakeet, whisper, and TTS can coexist on one device sharing a single ggml. It
carries patches (backend-registry filename prefix, OpenCL non-Adreno support,
OpenCL program-binary cache) and adds an explicit **load-time backend cascade
with fallbacks**, dynamic `.so` backend loading, and prewarm:

```cpp
// EngineOptions knobs unique to qvac's deployment model
opts.backends_dir     = appNativeLibDir;  // load libspeech-ggml-vulkan.so etc. (Android APK)
opts.opencl_cache_dir = appCacheDir;      // persist clBuildProgram blobs (Adreno cold-start)
opts.prewarm          = true;             // pay GPU pipeline compile at construction
// post-fallback truth (Adreno-tier policy / extension probe may force CPU):
engine.backend_device();  // CPU or GPU
```

---

## 6. Platform & GPU support matrix

CPU is available everywhere; **bold** = GPU acceleration.

| Platform / Arch | qvac | mudler |
|-----------------|------|--------|
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

---

## 7. Quantization formats

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

---

## 8. Checkpoint coverage

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

## 9. Feature differences

### Only in qvac (`transcription-parakeet`)
- **Speaker diarization** — Sortformer v1 / v2 / v2.1 with NeMo Audio-Online Speaker Cache (AOSC), so speakers rebind to their original slot across long gaps.
- **Speaker-attributed transcription** ("who said what") — ASR + Sortformer combined (`transcribe_with_speakers`, `live-mic-attributed`).
- **Live duplex streaming + microphone** — Mode 3 cache-aware chunks, `<EOU>` turn boundaries, `StreamEvent` callbacks, energy VAD, `live-mic` apps.
- **Mobile & embedded reach** — iOS and Android (arm64) builds, plus the **OpenCL** backend for Adreno GPUs.
- **Runtime integration** — Bare/Node native addon driven from the QVAC SDK (JS API, P2P), sharing one `qvac-speech-` ggml with other QVAC speech models.

### Only in mudler (`parakeet.cpp`)
- **CUDA (NVIDIA) and HIP/ROCm (AMD) backends** — qvac is Metal / Vulkan / OpenCL only.
- **K-quants** (`q4_k`, `q5_k`, `q6_k`) via `parakeet-cli quantize`.
- **More & larger checkpoints** — 1.1B family (CTC / RNNT / TDT / hybrid TDT+CTC), 110M hybrid, RNNT 0.6B, and **nemotron-3.5 streaming multilingual**.
- **Batched decode** (`bench-batch`, `--batch-sizes`) and a `bench-decode` microbenchmark.
- **Distribution surface** — flat C-API + shared lib for dlopen/FFI/LocalAI, prebuilt CLI binaries for 5 platforms, and Docker images (CPU + CUDA) on GHCR.
- **Word/segment timestamps** (`--timestamps`).

### Shared by both
CTC + TDT + EOU transcription · `q8_0` / `f16` · CPU + Metal + Vulkan · ggml-based ·
log-mel front-end on GPU · WER-0 parity vs NeMo on clean English.

---

## 10. Model types (decoders)

| Model | Full name | How it works | Trade-off | Languages |
|-------|-----------|--------------|-----------|-----------|
| **CTC** | Connectionist Temporal Classification | Non-autoregressive: predicts one token per audio frame (plus a "blank"), then collapses repeats/blanks into text in a single pass. | Fastest & simplest; no explicit duration model, slightly weaker on hard audio. | English |
| **TDT** | Token-and-Duration Transducer (RNN-T family) | Predicts each token *and how many frames to skip* (its duration), striding over audio instead of stepping frame-by-frame. | Best accuracy + punctuation/capitalization; multilingual. Slightly heavier decoder. | ~25 (v3) |
| **EOU** | End-of-Utterance streaming (RNN-T + `<EOU>`) | A small 120M streaming model that also emits an `<EOU>` token to detect when a speaker finished their turn. | Built for low-latency live conversation, not peak accuracy. | English |

Other decoders in the ecosystem: **RNNT** (plain transducer), **hybrid TDT+CTC**
(one checkpoint, both heads), and **Sortformer** (speaker *diarization*, not
transcription).

---

## Summary

- **qvac** is the broader *product* engine: one stateful `Engine`, diarization,
  speaker attribution, live streaming, and mobile (iOS/Android/OpenCL) reach,
  wired into the QVAC SDK on a shared cross-platform ggml.
- **mudler** is the broader *standalone* engine: a granular component-graph
  re-implementation with more checkpoints (1.1B, hybrid, RNNT, nemotron),
  K-quants, CUDA/HIP, batching, and a C-API/Docker distribution surface — but no
  diarization and no mobile.
