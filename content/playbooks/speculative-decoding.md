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

Speculative decoding uses a small, fast "draft" model to generate candidate tokens, which a larger "target" model verifies in a single forward pass. Accepted tokens are essentially "free" computation.

## Expected speedup

- Typical acceptance rate: 60–80%
- Typical speedup: **1.5–2.5×** on compatible model pairs
- Best with same-family models (e.g., Qwen2.5-1.5B draft + Qwen2.5-32B target)

## Prerequisites

- llama.cpp installed (see llama.cpp playbook)
- Two GGUF model files: a large target model and a small draft model

## Time & risk

- **Duration:** 15 minutes
- **Risk level:** Low

<!-- tab: Setup -->
## Download model pair

```bash
# Target model (32B)
huggingface-cli download bartowski/Qwen2.5-32B-Instruct-GGUF \
  --include "Qwen2.5-32B-Instruct-Q4_K_M.gguf" \
  --local-dir ~/models

# Draft model (1.5B — same family for best acceptance rate)
huggingface-cli download bartowski/Qwen2.5-1.5B-Instruct-GGUF \
  --include "Qwen2.5-1.5B-Instruct-Q8_0.gguf" \
  --local-dir ~/models
```

<!-- tab: Run -->
## Run with speculative decoding

```bash
llama-speculative \
  -m ~/models/Qwen2.5-32B-Instruct-Q4_K_M.gguf \
  -md ~/models/Qwen2.5-1.5B-Instruct-Q8_0.gguf \
  -ngl 99 -ngld 99 \
  -c 4096 \
  --draft-max 8 \
  -p "Write a detailed analysis of transformer architectures."
```

## Key flags

- `-m` — target (large) model path
- `-md` — draft (small) model path
- `-ngl 99` — offload target model layers to Metal GPU
- `-ngld 99` — offload draft model layers to Metal GPU
- `--draft-max 8` — number of candidate tokens per step

<!-- tab: Benchmarking -->
## Measure speedup

Run the same prompt with and without speculative decoding:

```bash
# Without speculative decoding
time llama-cli -m ~/models/Qwen2.5-32B-Instruct-Q4_K_M.gguf \
  -ngl 99 -p "Write 500 words about Apple Silicon." -n 500

# With speculative decoding
time llama-speculative \
  -m ~/models/Qwen2.5-32B-Instruct-Q4_K_M.gguf \
  -md ~/models/Qwen2.5-1.5B-Instruct-Q8_0.gguf \
  -ngl 99 -ngld 99 --draft-max 8 \
  -p "Write 500 words about Apple Silicon." -n 500
```

The output includes `accepted draft tokens` stats — higher acceptance = more speedup.
