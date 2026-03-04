---
slug: speculative-decoding
title: "Speculative Decoding"
time: "15 min"
color: green
desc: "Use draft models to accelerate generation 1.5–2.5×"
tags: [inference, optimization]
spark: "Speculative Decoding"
category: inference
featured: false
whatsNew: false
---

<!-- tab: Overview -->
## Basic idea

Speculative decoding is a technique where a small "draft" model proposes multiple tokens in one shot, then the large "target" model verifies all of them in a single forward pass. Tokens that pass verification are accepted for free; the first rejected token triggers a correction and the process repeats.

The key insight: verifying N tokens costs the same compute as generating 1 token, because a transformer's forward pass processes all positions in parallel. If the draft model proposes 8 tokens and 6 are accepted, you generated 6 tokens for the price of 1 verification. With a 60–80% acceptance rate, real-world speedups of 1.5–2.5× are typical — with **identical output quality** to running the target model alone.

## What you'll accomplish

A quantified speedup benchmark comparing standard inference against speculative decoding on the same prompt, using llama.cpp's `llama-speculative` binary with a Qwen2.5-32B target model and a Qwen2.5-1.5B draft model. You'll measure tokens/sec in both modes and calculate the actual multiplier on your hardware.

## What to know before starting

- **Transformer forward passes** — each pass generates one token's probability distribution over the vocabulary. The expensive part is the matrix multiplications across all layers, which run in parallel regardless of context length.
- **Draft-target alignment** — the draft model must be from the same model family as the target (same tokenizer, same pre-training distribution) for acceptance rates to be high. A Qwen2.5-1.5B draft with a Qwen2.5-32B target achieves 65–80% acceptance; a mismatched draft might get 20–30%.
- **Why acceptance rate matters** — at 100% acceptance with 8 draft tokens, you'd get 8× speedup. At 0% acceptance, every draft guess is wrong and you actually run slower than baseline due to wasted computation. The break-even acceptance rate is roughly 30%.
- **GGUF format** — the binary format llama.cpp uses for quantized models. Each file contains model weights, tokenizer, and metadata. GGUF files are self-contained and portable.
- **GPU offloading with -ngl** — "number of GPU layers" controls how many transformer layers are computed on the Metal GPU vs CPU RAM. -ngl 99 offloads all layers, maximizing throughput.

## Prerequisites

- llama.cpp installed with `llama-speculative` binary in your PATH (see the llama.cpp playbook)
- Two GGUF model files — same model family, different sizes (see Setup tab for exact downloads)
- 32 GB+ unified memory (the 32B Q4_K_M target needs ~20 GB, the 1.5B draft needs ~1.7 GB; both must fit simultaneously)
- `huggingface-cli` for downloading models: `pip install huggingface_hub`

## Time & risk

- **Duration:** 15 minutes setup (plus model download time — ~20 GB total on first run)
- **Risk level:** Low — read-only model inference, no system modifications
- **Rollback:** Nothing to roll back; simply stop the process

<!-- tab: Setup -->
## Step 1: Verify llama-speculative is installed

The `llama-speculative` binary ships with llama.cpp but is a separate executable from `llama-cli`. Confirm it exists before downloading large model files.

```bash
which llama-speculative           # should print a path like /opt/homebrew/bin/llama-speculative
llama-speculative --version       # prints the llama.cpp build version
```

If the command is not found, your llama.cpp installation may be outdated. Run `brew upgrade llama.cpp` or reinstall from the llama.cpp playbook. The speculative binary was added in build b2359.

## Step 2: Download the draft model

The draft model needs to be fast and from the same family as the target. Qwen2.5-1.5B at Q8_0 quantization is the right choice: Q8_0 is near-lossless (only 0.1% perplexity increase), which keeps the draft model's predictions closely aligned with what the 32B target would predict. A lower-quality draft = lower acceptance rate = less speedup.

```bash
huggingface-cli download bartowski/Qwen2.5-1.5B-Instruct-GGUF \
  --include "Qwen2.5-1.5B-Instruct-Q8_0.gguf" \    # Q8_0: near-lossless, ~1.7 GB
  --local-dir ~/models                               # saves to ~/models/
```

Expected file size: ~1.7 GB. Download time: 2–5 minutes on a fast connection.

## Step 3: Download the target model

The target model is the 32B parameter Qwen2.5-Instruct at Q4_K_M quantization. Q4_K_M uses 4-bit integers with K-means clustering for improved accuracy over naive Q4 — it's the standard recommended quantization for general use.

```bash
huggingface-cli download bartowski/Qwen2.5-32B-Instruct-GGUF \
  --include "Qwen2.5-32B-Instruct-Q4_K_M.gguf" \   # Q4_K_M: best 4-bit quality, ~19.8 GB
  --local-dir ~/models                               # both models in same directory
```

Expected file size: ~19.8 GB. Download time: 10–30 minutes depending on connection. Both models must be loaded into unified memory simultaneously, which is why 32 GB+ is required.

## Step 4: Verify both models load independently

Before running speculative decoding, test each model alone. This confirms the files are not corrupted and helps you establish the baseline tokens/sec for the target model.

```bash
# Test the draft model loads and generates output
llama-cli \
  -m ~/models/Qwen2.5-1.5B-Instruct-Q8_0.gguf \    # draft model
  -ngl 99 \                                          # offload all layers to Metal GPU
  -n 50 \                                            # generate 50 tokens
  -p "Hello, tell me about yourself." \
  --log-disable                                      # suppress verbose Metal logs

# Test the target model loads and generates output
llama-cli \
  -m ~/models/Qwen2.5-32B-Instruct-Q4_K_M.gguf \   # target model
  -ngl 99 \
  -n 50 \
  -p "Hello, tell me about yourself." \
  --log-disable
```

Look for `llama_print_timings: eval time` in the output — this shows tokens/sec. Note the target model's speed. If either model fails to load, check the file size matches expected (a partial download will cause a "magic number" error).

<!-- tab: Run -->
## Step 1: Run with speculative decoding

The `llama-speculative` command combines both models. Every flag matters for performance — here's what each one does and why it's set this way.

```bash
llama-speculative \
  -m ~/models/Qwen2.5-32B-Instruct-Q4_K_M.gguf \   # -m: the target (large, verifier) model
  -md ~/models/Qwen2.5-1.5B-Instruct-Q8_0.gguf \   # -md: the draft (small, proposer) model
  -ngl 99 \                                          # offload ALL target model layers to Metal GPU
  -ngld 99 \                                         # offload ALL draft model layers to Metal GPU
  --draft-max 8 \                                    # propose 8 tokens per draft step (tune this!)
  -c 4096 \                                          # context window (must match between models)
  -p "Write a detailed technical analysis of Apple Silicon's unified memory architecture and why it matters for machine learning workloads." \
  -n 400                                             # generate 400 tokens for a meaningful benchmark
```

Both `-ngl` and `-ngld` must be set — omitting `-ngld` leaves the draft model on CPU, making it slow enough to cancel out the speculative speedup.

## Step 2: Read the acceptance rate stats

After generation completes, llama.cpp prints timing statistics. The critical line is `draft acceptance`:

```
llama_print_timings:        load time =   2847.23 ms
llama_print_timings:      sample time =     18.45 ms
llama_print_timings: prompt eval time =    842.11 ms
llama_print_timings:        eval time =  12304.56 ms / 400 runs (  30.76 ms per token,  32.51 tokens/s)
draft acceptance:  267/400 accepted, 66.8%
```

Interpreting acceptance rate:

| Acceptance Rate | Assessment | Action |
|---|---|---|
| > 75% | Excellent — near-theoretical speedup | Increase `--draft-max` to 10-12 |
| 60–75% | Good — expect 1.8–2.3× speedup | Default settings are well-tuned |
| 40–60% | Moderate — some speedup, diminishing returns | Consider `--draft-max 4-6` |
| < 40% | Poor — draft model misaligned | Check model family match |

## Step 3: Tune draft-max for your workload

The `--draft-max` value controls how many tokens the draft model proposes before the target verifies. Larger values increase the potential speedup but also increase the probability that the chain breaks (one wrong token invalidates all subsequent ones).

```bash
# Conservative: 4 draft tokens — better for complex reasoning tasks
llama-speculative -m ~/models/Qwen2.5-32B-Instruct-Q4_K_M.gguf \
  -md ~/models/Qwen2.5-1.5B-Instruct-Q8_0.gguf \
  -ngl 99 -ngld 99 --draft-max 4 -c 4096 \
  -p "Explain Bayesian inference step by step." -n 300

# Aggressive: 12 draft tokens — better for repetitive or predictable text
llama-speculative -m ~/models/Qwen2.5-32B-Instruct-Q4_K_M.gguf \
  -md ~/models/Qwen2.5-1.5B-Instruct-Q8_0.gguf \
  -ngl 99 -ngld 99 --draft-max 12 -c 4096 \
  -p "Write a Python script that reads a CSV file." -n 300
```

As a rule of thumb: code generation and repetitive tasks benefit from higher `--draft-max`; creative writing and reasoning tasks do better with 6–8.

<!-- tab: Benchmarking -->
## Step 1: Baseline measurement without speculative decoding

Run the target model alone and record its tokens/sec. Use `--log-disable` to suppress Metal initialization logs and focus on the timing output. The `-n 300` ensures enough tokens for a stable average.

```bash
# Baseline: standard single-model inference
llama-cli \
  -m ~/models/Qwen2.5-32B-Instruct-Q4_K_M.gguf \
  -ngl 99 \                                    # GPU offload
  -c 4096 \                                    # same context as speculative run
  -n 300 \                                     # generate 300 tokens for stable measurement
  -p "Write 300 words about the history of machine learning." \
  --log-disable 2>&1 | grep "eval time"        # extract just the timing line
```

Record the `tokens/s` value from the `eval time` line. This is your baseline.

## Step 2: Measurement with speculative decoding

Run the same prompt with `llama-speculative` at `--draft-max 8`. Keep all other parameters identical to the baseline so the comparison is valid.

```bash
# Speculative: draft model assists target model
llama-speculative \
  -m ~/models/Qwen2.5-32B-Instruct-Q4_K_M.gguf \
  -md ~/models/Qwen2.5-1.5B-Instruct-Q8_0.gguf \
  -ngl 99 -ngld 99 \
  --draft-max 8 \
  -c 4096 \
  -n 300 \
  -p "Write 300 words about the history of machine learning." \
  --log-disable 2>&1 | grep -E "eval time|draft acceptance"
```

Record both the `tokens/s` and the `draft acceptance` percentage.

## Step 3: Calculate and interpret speedup

Divide speculative tokens/sec by baseline tokens/sec:

```
speedup = speculative_tps / baseline_tps

Example: 48.3 tps / 22.1 tps = 2.18× speedup
```

## Reference results

These are representative numbers on Apple Silicon. Your results will vary based on RAM configuration and thermal state.

| Hardware | RAM | Baseline (32B Q4_K_M) | Speculative (draft-max 8) | Speedup | Acceptance |
|---|---|---|---|---|---|
| M2 Ultra | 192 GB | 28–32 t/s | 58–68 t/s | ~2.1× | 72% |
| M3 Max | 128 GB | 22–26 t/s | 44–54 t/s | ~2.0× | 69% |
| M1 Pro | 32 GB | 8–12 t/s | 14–20 t/s | ~1.7× | 63% |

The M1 Pro shows lower speedup because unified memory bandwidth is the bottleneck — both models compete for the same bus, slightly reducing the theoretical benefit. On M2 Ultra and M3 Max, the higher memory bandwidth allows both models to run efficiently in parallel.

<!-- tab: Troubleshooting -->
## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Low acceptance rate (< 40%) | Draft and target are different model families | Use same-family models: Qwen2.5-1.5B with Qwen2.5-32B, not Llama draft with Qwen target |
| OOM crash on launch | Both models don't fit in unified memory | Try Q3_K_M for target (~15 GB) or use 64 GB machine |
| `llama-speculative: command not found` | Binary not in PATH or old llama.cpp build | Run `brew upgrade llama.cpp` or check installation with `brew list llama.cpp` |
| Draft model generation slower than baseline | `-ngld` flag missing, draft running on CPU | Add `-ngld 99` to offload draft model layers to Metal GPU |
| Metal allocation failure on load | VRAM fragmentation from prior processes | Restart the terminal session or run `sudo purge` to clear memory caches |
| Context mismatch error | `-c` value differs between invocations | Use identical `-c 4096` in both baseline and speculative commands |
| Tokenizer mismatch warning | Draft and target use different tokenizers | Only use models from exact same tokenizer family — check the HF model card |

### Diagnosing low acceptance rates

If acceptance is under 50%, run this diagnostic to isolate whether it's a model family issue or a prompt type issue:

```bash
# Test with a predictable prompt (should get high acceptance)
llama-speculative -m ~/models/Qwen2.5-32B-Instruct-Q4_K_M.gguf \
  -md ~/models/Qwen2.5-1.5B-Instruct-Q8_0.gguf \
  -ngl 99 -ngld 99 --draft-max 8 -c 4096 \
  -p "Count from 1 to 20: 1, 2, 3," -n 60

# Test with a complex prompt (should get lower acceptance)
llama-speculative -m ~/models/Qwen2.5-32B-Instruct-Q4_K_M.gguf \
  -md ~/models/Qwen2.5-1.5B-Instruct-Q8_0.gguf \
  -ngl 99 -ngld 99 --draft-max 8 -c 4096 \
  -p "Derive the Euler-Lagrange equations from first principles." -n 200
```

If the counting prompt gets 80%+ and the math prompt gets 50%, the draft model is working correctly — the 32B model diverges from the 1.5B prediction on complex reasoning. Reduce `--draft-max` to 4 for complex prompts.

### Handling Metal allocation failures

Apple Silicon's unified memory is managed by the Metal memory allocator. When multiple GPU-intensive processes run simultaneously, memory can become fragmented.

```bash
# Check current memory pressure
memory_pressure                                      # shows free/wired/compressed/active

# Free up Metal memory caches (requires sudo)
sudo purge

# Then retry with slightly reduced offload (leave 2 layers on CPU)
llama-speculative -m ~/models/Qwen2.5-32B-Instruct-Q4_K_M.gguf \
  -md ~/models/Qwen2.5-1.5B-Instruct-Q8_0.gguf \
  -ngl 62 -ngld 99 \                               # keep top 2 target layers on CPU
  --draft-max 8 -c 4096 -p "Your prompt here." -n 200
```

### Understanding the timing output in detail

```
llama_print_timings: prompt eval time = 842 ms / 52 tokens  # time to process your input prompt
llama_print_timings:        eval time = 12304 ms / 400 runs # total generation time
# The "runs" count equals tokens generated — each run is one verification step
# Divide eval_time by runs to get ms/token, then invert for tokens/sec
```

The speculative run's "runs" count equals the number of verification calls, not the number of tokens generated. With 66% acceptance and draft-max 8, roughly 1 verification call produces ~5.3 tokens on average.
