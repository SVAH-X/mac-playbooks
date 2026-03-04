---
slug: mlx-quantization
title: "MLX Quantization"
time: "10 min"
color: green
desc: "Quantize any HF model to 2/4/8-bit for Apple Silicon"
tags: [mlx, quantization]
spark: "NVFP4 Quantization"
category: inference
featured: false
whatsNew: false
---

<!-- tab: Overview -->
## Basic idea

MLX can quantize any Hugging Face model to 2, 3, 4, or 8-bit precision, dramatically reducing memory requirements while preserving most quality. Quantized models are stored locally and run faster than their full-precision counterparts.

## Prerequisites

- macOS 14.0+
- Apple Silicon Mac
- Python 3.10+
- `mlx-lm` installed
- Sufficient disk space (model size × 2 during conversion)

## Time & risk

- **Duration:** 10 minutes setup, plus conversion time (varies by model size)
- **Risk level:** None — creates new files, doesn't modify originals

<!-- tab: Quantize -->
## Quantize to 4-bit (recommended default)

```bash
mlx_lm.convert \
  --hf-path Qwen/Qwen2.5-7B-Instruct \
  --mlx-path ./qwen2.5-7b-4bit \
  --quantize \
  --q-bits 4
```

## Quantize to 8-bit (near-lossless)

```bash
mlx_lm.convert \
  --hf-path Qwen/Qwen2.5-7B-Instruct \
  --mlx-path ./qwen2.5-7b-8bit \
  --quantize \
  --q-bits 8
```

## Quantize to 2-bit (maximum compression)

```bash
mlx_lm.convert \
  --hf-path Qwen/Qwen2.5-32B-Instruct \
  --mlx-path ./qwen2.5-32b-2bit \
  --quantize \
  --q-bits 2
```

## Run the quantized model

```bash
mlx_lm.generate --model ./qwen2.5-7b-4bit --prompt "Hello!"
```

<!-- tab: Guidelines -->
## Quantization quality vs memory tradeoffs

| Bits | Quality | Memory Savings | Best For |
|---|---|---|---|
| 8-bit | Near-lossless | ~50% | Quality-critical tasks |
| 4-bit | Good | ~75% | General use (recommended default) |
| 3-bit | Acceptable | ~81% | Memory-constrained machines |
| 2-bit | Noticeable degradation | ~87% | Fitting large models in limited RAM |

## Pre-quantized models

Many pre-quantized models are available in the `mlx-community` organization on Hugging Face — no conversion needed:

```bash
mlx_lm.generate --model mlx-community/Qwen2.5-32B-Instruct-4bit --prompt "Hello!"
```

This is the fastest way to get started.

<!-- tab: Troubleshooting -->
## Conversion runs out of memory

The conversion process loads the full-precision model. Ensure you have at least 2× the model size in free memory.

For very large models, use a machine with more RAM or convert on a cloud instance.

## HuggingFace authentication required

Some gated models require authentication:
```bash
huggingface-cli login
```

Then re-run the conversion command.
