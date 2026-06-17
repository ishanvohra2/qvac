# Parakeet engine comparison: qvac vs `mudler/parakeet.cpp`

Benchmarks the qvac **`parakeet-cpp` C++ engine** (from
[`tetherto/qvac-ext-lib-whisper.cpp`](https://github.com/tetherto/qvac-ext-lib-whisper.cpp),
the engine the `transcription-parakeet` addon wraps) against
[`mudler/parakeet.cpp`](https://github.com/mudler/parakeet.cpp) (the `parakeet-cli`
binary) on the **same** checkpoints, **same** quant level (`q8_0`), and **same**
canonical 16 kHz mono clips.

Per model × backend (CPU / GPU) × clip it records:

- **processing time** and **RTF** (`proc / audio`, lower is faster) — warmup + N timed reps
- **WER** against ground-truth transcripts (English clips + FLEURS multilingual corpus)
- cross-engine agreement where no reference exists

The harness is platform-aware: it auto-names its outputs
`report-<platform>.{md,html}` (e.g. `report-mac-arm64.md`, `report-linux-x64.md`)
so a Mac run and an NVIDIA run never clobber each other. A committed Mac
(Apple Silicon, Metal) baseline lives in [`baseline/mac-arm64/`](baseline/mac-arm64/).

> **Engine-to-engine, no JS tax.** Both timings are engine-only C++ inference
> (mel + encoder + decoder), excluding model load and wav read: qvac =
> `parakeet-cpp --bench` (`inference_ms`); mudler = `parakeet-cli bench`
> (`transcribe_pcm`). We deliberately do **not** time the qvac Bare/Node addon
> here — that would add JS marshalling overhead and isn't comparable to mudler's
> engine-only number. Each engine loads its **own native** `q8_0` GGUF — the two
> GGUF schemas are not interchangeable (different tensor names + KV metadata).

---

## Files

| File | Role |
|------|------|
| `run-comparison.js` | Orchestrator: builds clips, runs both engine CLIs (`--bench`), renders `.md` + `.html` + `.json`. |
| `fetch-fleurs.js` | Downloads a small labelled FLEURS subset (fr/es/hr) for real multilingual WER. |
| `qvac-bench.js` | Legacy Bare/addon driver — **no longer used** (kept for reference; the harness now benchmarks the `parakeet-cpp` engine CLI directly). |
| `baseline/<platform>/` | Committed reference reports (Mac baseline included). |
| `out/` | Generated working dir (clips, FLEURS corpus ~600 MB, per-engine JSON, live reports). **gitignored.** |

---

## Prerequisites

- Node.js (orchestrator) on `PATH`.
- A C/C++ toolchain + CMake ≥ 3.20 for building both engines.
- `git` (use `gh` if `https` clones prompt for credentials), `curl`, and `tar`.

---

## Step 1 — Stage the qvac models

From `packages/transcription-parakeet` (`npm install` is only needed for the
registry download client; the addon build itself is not required):

```bash
npm install
npm run download-models:registry -- --type tdt,ctc,eou --quant q8_0
```

This puts the qvac GGUFs in `packages/transcription-parakeet/models/`:

- `parakeet-tdt-0.6b-v3.q8_0.gguf` (multilingual)
- `parakeet-ctc-0.6b.q8_0.gguf` (English)
- `parakeet-eou-120m-v1.q8_0.gguf` (streaming, English)

## Step 2 — Build qvac's `parakeet-cpp` engine CLI

The qvac side is benchmarked via its engine CLI (binary `parakeet`), built from
the `parakeet-cpp` subdir of `tetherto/qvac-ext-lib-whisper.cpp`. Clone it as a
sibling of this package (gitignored):

```bash
cd packages
gh repo clone tetherto/qvac-ext-lib-whisper.cpp        # or: git clone <url>
cd qvac-ext-lib-whisper.cpp/parakeet-cpp

# Fetch the pinned ggml (speech branch). setup-ggml.sh clones over https; if
# that prompts for credentials on your machine, clone ggml via gh instead:
gh repo clone tetherto/qvac-ext-ggml ggml -- --branch speech --depth 1
# (or: ./scripts/setup-ggml.sh)
```

Then configure with **one** GPU backend (matching the platform under test):

```bash
# macOS (Apple Silicon, Metal)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
  -DPARAKEET_BUILD_TESTS=OFF -DPARAKEET_BUILD_EXAMPLES=OFF
cmake --build build -j --target parakeet-cli

# NVIDIA / desktop (Vulkan) — qvac has no CUDA backend
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON \
  -DPARAKEET_BUILD_TESTS=OFF -DPARAKEET_BUILD_EXAMPLES=OFF
cmake --build build -j --target parakeet-cli
```

The harness looks for the binary at
`packages/qvac-ext-lib-whisper.cpp/parakeet-cpp/build/parakeet` (it also probes
`build-metal/` / `build-vk/`). Override with `QVAC_PARAKEET_CLI=/path/to/parakeet`
or point at a different clone with `QVAC_ENGINE_DIR=...`.

## Step 3 — Clone and build `mudler/parakeet.cpp`

Cloned into `packages/parakeet.cpp` (sibling of this package; gitignored).

```bash
cd packages
git clone --recursive https://github.com/mudler/parakeet.cpp
cd parakeet.cpp
```

**macOS (Apple Silicon, Metal):**

```bash
cmake -B build -DPARAKEET_GGML_METAL=ON
cmake --build build -j
```

**NVIDIA (Linux/Windows, CUDA):**

```bash
cmake -B build -DPARAKEET_GGML_CUDA=ON
cmake --build build -j
```

> CUDA bundles target Turing (sm_75) and newer (incl. Blackwell / GB10).
> For an NVIDIA box without CUDA you can instead use `-DPARAKEET_GGML_VULKAN=ON`
> (needs the Vulkan loader). The CLI auto-selects the first GPU device; set
> `PARAKEET_DEVICE=cpu` to force CPU (the harness does this for CPU rows).
>
> **For an apples-to-apples GPU comparison, build mudler with the same backend
> as qvac.** qvac has no CUDA backend (Metal / Vulkan / OpenCL only), so on
> NVIDIA prefer `-DPARAKEET_GGML_VULKAN=ON` for both — otherwise the GPU column
> compares CUDA (mudler) vs Vulkan (qvac), which is a backend difference as much
> as an engine one.

The harness expects the binary at `packages/parakeet.cpp/build/examples/cli/parakeet-cli`.

## Step 4 — Get mudler's GGUFs

mudler publishes its converted GGUFs in the HF collection repo
[`mudler/parakeet-cpp-gguf`](https://huggingface.co/mudler/parakeet-cpp-gguf).
Download the `q8_0` files into `packages/parakeet.cpp/models-gguf/` with these
**exact** names (this is what the harness looks for):

- `tdt-0.6b-v3-q8_0.gguf`
- `ctc-0.6b-q8_0.gguf`
- `realtime_eou_120m-v1-q8_0.gguf`

```bash
cd packages/parakeet.cpp
mkdir -p models-gguf
huggingface-cli download mudler/parakeet-cpp-gguf <file> --local-dir models-gguf
# ...or download each file from the repo's web UI and rename to match above.
```

> Why not reuse the qvac GGUF? The two `q8_0` writers keep slightly different
> tensor sets and use different tensor names + KV metadata, so each CLI can only
> load its own file. Same checkpoint, same quant level — different container.

## Step 5 — (Optional) FLEURS multilingual WER

Real ground-truth WER for fr/es/hr (TDT only). Downloads ~600 MB into `out/fleurs`.

```bash
cd packages/transcription-parakeet
node benchmarks/comparison/fetch-fleurs.js
```

Skip this and the harness simply omits section 4.

## Step 6 — Run the comparison

```bash
cd packages/transcription-parakeet
node benchmarks/comparison/run-comparison.js
```

Outputs land in `benchmarks/comparison/out/`:

- `report-<platform>.md`
- `report-<platform>.html`
- `comparison-data-<platform>.json` (raw data incl. qvac's actual `backendId`)

---

## Configuration (env vars)

| Var | Default | Meaning |
|-----|---------|---------|
| `QVAC_CMP_MODELS` | `tdt,ctc,eou` | Which models to run. |
| `QVAC_CMP_GPU` | both | `true` = GPU only, `false` = CPU only, unset = both. |
| `QVAC_CMP_RUNS` | `5` | Timed reps per clip. |
| `QVAC_CMP_WARMUP` | `1` | Warmup reps (discarded). |
| `QVAC_CMP_THREADS` | `4` | Threads for both engines. |
| `QVAC_CMP_QUANT` | `q8_0` | Quant label (reporting only). |
| `QVAC_PARAKEET_CLI` | probes `build*/parakeet` | Path to qvac's built `parakeet` engine binary. |
| `QVAC_ENGINE_DIR` | `packages/qvac-ext-lib-whisper.cpp/parakeet-cpp` | Root of the qvac engine clone (used to find the binary). |
| `QVAC_CMP_SKIP_MATRIX` | – | `1` reuses the cached matrix and only re-renders / re-runs FLEURS. |
| `QVAC_CMP_FLEURS_GPU` | `true` | `false` runs the FLEURS pass on CPU. |
| `QVAC_CMP_GPU_LABEL` | `Metal` on macOS, else `GPU` | Display label for the GPU backend column (e.g. `Vulkan`, `CUDA`). |
| `QVAC_CMP_PLATFORM_NOTE` | derived | Free-text note shown in the report header. |
| `QVAC_CMP_RENDER_FROM` | – | Path to an existing `comparison-data-*.json`; re-renders just the `.md`/`.html` (no engine runs) using that file's platform. Useful after editing report templates. |

Example (CPU + GPU, TDT only, NVIDIA, labelled "CUDA"):

```bash
QVAC_CMP_MODELS=tdt QVAC_CMP_GPU_LABEL=CUDA \
  node benchmarks/comparison/run-comparison.js
```

---

## Notes on cross-platform results

- The **Backend** column reads `CPU` / `<GPU_LABEL>`. On Apple Silicon both
  engines use Metal. On NVIDIA the GPU row means **CUDA for mudler** and
  **Vulkan for qvac** (qvac has no CUDA backend); qvac's actual active backend
  is always recorded as `backendId` in `comparison-data-<platform>.json`.
- Sortformer diarization is **qvac-only** — mudler has no diarization, so it is
  excluded from this comparison.
- CTC and EOU are English-only; their non-English transcripts are expected to be
  wrong (timing is still valid). Only TDT is benchmarked for multilingual WER.
