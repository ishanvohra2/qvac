# Parakeet Engine Comparison: qvac vs mudler/parakeet.cpp

Generated: 2026-06-17T08:14:54.856Z  
Platform: `darwin-arm64` (Apple Silicon, Metal)  
Quant: `q8_0` · Threads: 4 · Warmup: 1 · Timed reps: 5

**RTF** = proc/audio (lower is faster) · **WER** lower is better.

> qvac time = full JS `run()` wall (product-level, includes Bare/JS tax); mudler time = engine-only `transcribe_pcm`. Same canonical clips, same threads, same quant level. Each engine loads its own native q8_0 GGUF (the two schemas are not interchangeable).

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

## Notes
- Sortformer diarization (v1 + v2.1 AOSC) is **qvac-only** — mudler has no diarization, excluded from this comparison.
- mudler-only (not benchmarked): K-quants, batched decode, 1.1B/RNNT/nemotron multilingual streaming, CUDA/HIP.
- CTC and EOU are English-only; their transcripts on non-English clips are expected to be wrong (timing still valid).
- GGUF schemas are not interchangeable (verified both directions): qvac uses renamed `blk`-style tensors + `parakeet.*` KV; mudler keeps verbatim NeMo names. Each engine runs its own native q8_0 file.
