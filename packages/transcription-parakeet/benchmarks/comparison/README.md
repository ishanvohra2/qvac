# Parakeet engine comparison: qvac vs `mudler/parakeet.cpp`

Benchmarks the qvac `transcription-parakeet` ggml addon against
[`mudler/parakeet.cpp`](https://github.com/mudler/parakeet.cpp) (the `parakeet-cli`
binary) on the **same** checkpoints, **same** quant level (`q8_0`), and **same**
canonical 16 kHz mono clips.

Per model × backend (CPU / GPU) × clip it records:

- **processing time** and **RTF** (`proc / audio`, lower is faster) — load-once + warmup + N timed reps
- **WER** against ground-truth transcripts (English clips + FLEURS multilingual corpus)
- cross-engine agreement where no reference exists

The harness is platform-aware: it auto-names its outputs
`report-<platform>.{md,html}` (e.g. `report-mac-arm64.md`, `report-linux-x64.md`)
so a Mac run and an NVIDIA run never clobber each other. A committed Mac
(Apple Silicon, Metal) baseline lives in [`baseline/mac-arm64/`](baseline/mac-arm64/).

> Heads up on fairness: qvac time is the full JS `run()` wall (product-level,
> includes a small Bare/JS tax); mudler time is engine-only `transcribe_pcm`
> from its `bench` command. Each engine loads its **own native** `q8_0` GGUF —
> the two GGUF schemas are not interchangeable (different tensor names + KV
> metadata), so you cannot share one file between them.

---

## Files

| File | Role |
|------|------|
| `run-comparison.js` | Orchestrator: builds clips, runs both engines, renders `.md` + `.html` + `.json`. |
| `qvac-bench.js` | Bare driver for the qvac addon (load once, warmup, timed reps, transcript). |
| `fetch-fleurs.js` | Downloads a small labelled FLEURS subset (fr/es/hr) for real multilingual WER. |
| `baseline/<platform>/` | Committed reference reports (Mac baseline included). |
| `out/` | Generated working dir (clips, FLEURS corpus ~600 MB, per-engine JSON, live reports). **gitignored.** |

---

## Prerequisites

- Node.js (orchestrator) and the **Bare** runtime (`bare`) on `PATH`.
- A C/C++ toolchain + CMake for building `parakeet.cpp`.
- `git`, `curl`, and `tar` (used by `fetch-fleurs.js`).

---

## Step 1 — Build the qvac addon and stage its models

From `packages/transcription-parakeet`:

```bash
npm install
npm run build                  # bare-make generate && build && install
npm run download-models:registry -- --type tdt,ctc,eou --quant q8_0
```

This puts the qvac GGUFs in `packages/transcription-parakeet/models/`:

- `parakeet-tdt-0.6b-v3.q8_0.gguf` (multilingual)
- `parakeet-ctc-0.6b.q8_0.gguf` (English)
- `parakeet-eou-120m-v1.q8_0.gguf` (streaming, English)

> Already have prebuilds + models staged? Skip the build/download.

## Step 2 — Clone and build `mudler/parakeet.cpp`

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
> For an NVIDIA box without CUDA you can instead use `-DGGML_VULKAN=ON`
> (needs the Vulkan loader). The CLI auto-selects the first GPU device; set
> `PARAKEET_DEVICE=cpu` to force CPU (the harness does this for CPU rows).

The harness expects the binary at `packages/parakeet.cpp/build/examples/cli/parakeet-cli`.

## Step 3 — Get mudler's GGUFs

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

## Step 4 — (Optional) FLEURS multilingual WER

Real ground-truth WER for fr/es/hr (TDT only). Downloads ~600 MB into `out/fleurs`.

```bash
cd packages/transcription-parakeet
node benchmarks/comparison/fetch-fleurs.js
```

Skip this and the harness simply omits section 4.

## Step 5 — Run the comparison

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
