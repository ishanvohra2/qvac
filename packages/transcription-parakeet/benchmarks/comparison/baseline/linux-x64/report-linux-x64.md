# Parakeet Engine Comparison: qvac vs mudler/parakeet.cpp

> ⚠️ **Superseded — re-run pending.** This linux-x64 report was produced with the
> **old methodology**: the qvac side was measured through the Bare/Node
> `transcription-parakeet` **addon** (full JS `run()` wall, which includes
> JS/Bare runtime overhead), *not* the standalone `parakeet-cpp` engine CLI. The
> harness has since moved to an **engine-to-engine** comparison (qvac
> `parakeet-cpp --bench` vs mudler `parakeet-cli bench`), as used in the
> `mac-arm64` baseline. Regenerate this report on the NVIDIA box with the updated
> harness (qvac `parakeet-cpp` CLI built for Vulkan) before citing these numbers.

Generated: 2026-06-17T10:14:59.387Z  
Platform: `linux-x64` (linux-x64, GPU)  
Quant: `q8_0` · Threads: 4 · Warmup: 1 · Timed reps: 5

**RTF** = proc/audio (lower is faster) · **WER** lower is better.

> qvac time = full JS `run()` wall (product-level, includes Bare/JS tax); mudler time = engine-only `transcribe_pcm`. Same canonical clips, same threads, same quant level. Each engine loads its own native q8_0 GGUF (the two schemas are not interchangeable).

## 1. Headline speed (clip: alice, English ~20.1s)

| Model | Backend | Engine | Proc ms | RTF | Faster |
|-------|---------|--------|--------:|----:|--------|
| TDT | CPU | qvac | 1265.9 | 0.0629 |  |
| TDT | CPU | mudler | 773.7 | 0.0384 | **mudler** 1.64x |
| TDT | GPU | qvac | 43.7 | 0.0022 | **qvac** 7.57x |
| TDT | GPU | mudler | 331.0 | 0.0164 |  |
| CTC | CPU | qvac | 1142.4 | 0.0567 |  |
| CTC | CPU | mudler | 715.8 | 0.0356 | **mudler** 1.60x |
| CTC | GPU | qvac | 21.2 | 0.0011 | **qvac** 1.37x |
| CTC | GPU | mudler | 29.1 | 0.0014 |  |
| EOU | CPU | qvac | 482.9 | 0.0240 |  |
| EOU | CPU | mudler | 329.6 | 0.0164 | **mudler** 1.46x |
| EOU | GPU | qvac | 57.0 | 0.0028 |  |
| EOU | GPU | mudler | 50.3 | 0.0025 | **mudler** 1.13x |

## 2. RTF vs clip duration (speed stability across lengths)

### TDT — CPU

| Clip | Lang | Duration s | qvac RTF | mudler RTF | Faster |
|------|------|-----------:|---------:|-----------:|--------|
| jfk | en | 11.0 | 0.0612 | 0.0363 | mudler 1.69x |
| alice | en | 20.1 | 0.0629 | 0.0384 | mudler 1.64x |
| croatian | hr | 27.4 | 0.0638 | 0.0392 | mudler 1.63x |
| french | fr | 29.4 | 0.0647 | 0.0382 | mudler 1.69x |
| spanish60 | es | 60.0 | 0.0712 | 0.0459 | mudler 1.55x |

### TDT — GPU

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

### CTC — GPU

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

### EOU — GPU

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
| TDT | GPU | jfk | en | 0.0% | 0.0% | 0.0% |
| TDT | GPU | alice | en | 0.0% | 0.0% | 0.0% |
| TDT | GPU | croatian | hr | n/a | n/a | 20.0% |
| TDT | GPU | french | fr | n/a | n/a | 34.4% |
| TDT | GPU | spanish60 | es | n/a | n/a | 46.6% |
| CTC | CPU | jfk | en | 0.0% | 0.0% | 0.0% |
| CTC | CPU | alice | en | 0.0% | 0.0% | 0.0% |
| CTC | GPU | jfk | en | 0.0% | 0.0% | 0.0% |
| CTC | GPU | alice | en | 0.0% | 0.0% | 0.0% |
| EOU | CPU | jfk | en | 0.0% | 4.5% | 4.5% |
| EOU | CPU | alice | en | 0.0% | 1.8% | 1.8% |
| EOU | GPU | jfk | en | 0.0% | 4.5% | 4.5% |
| EOU | GPU | alice | en | 0.0% | 1.8% | 1.8% |

## 4. Multilingual accuracy — FLEURS ground truth (TDT, GPU)

Real WER against FLEURS reference transcripts (corpus-level: total word edits / total reference words). TDT 0.6B v3 is the only multilingual model; CTC/EOU are English-only and excluded here.

| Language | Utterances | Ref words | qvac WER | mudler WER | Closer to reference |
|----------|-----------:|----------:|---------:|-----------:|---------------------|
| French (fr) | 12 | 265 | 12.8% | 7.2% | mudler |
| Spanish (es) | 12 | 306 | 1.6% | 1.0% | mudler |
| Croatian (hr) | 12 | 204 | 21.6% | 21.6% | qvac |

## Notes
- Sortformer diarization (v1 + v2.1 AOSC) is **qvac-only** — mudler has no diarization, excluded from this comparison.
- mudler-only (not benchmarked): K-quants, batched decode, 1.1B/RNNT/nemotron multilingual streaming, CUDA/HIP.
- CTC and EOU are English-only; their transcripts on non-English clips are expected to be wrong (timing still valid).
- GGUF schemas are not interchangeable (verified both directions): qvac uses renamed `blk`-style tensors + `parakeet.*` KV; mudler keeps verbatim NeMo names. Each engine runs its own native q8_0 file.
