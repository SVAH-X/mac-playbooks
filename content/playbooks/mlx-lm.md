---
slug: mlx-lm
title: "MLX LM for Inference"
time: "10 min"
color: green
desc: "Apple's high-performance native LLM engine — fastest on Mac"
tags: [mlx, inference]
spark: "vLLM for Inference"
category: inference
featured: true
whatsNew: false
---

<!-- tab: Overview -->
## Basic idea

MLX LM is Apple's purpose-built LLM inference engine for Apple Silicon. It uses the unified memory architecture for zero-copy CPU/GPU operations, achieving the highest throughput of any local inference runtime on Mac.

## Why MLX over vLLM on Mac?

- vLLM requires CUDA and does not run on macOS
- MLX is built from scratch for Apple Silicon's unified memory
- MLX achieves higher throughput than llama.cpp, Ollama, and PyTorch MPS
- Supports 2-bit to 8-bit quantization natively

## Prerequisites

- macOS 14.0+ (Sonoma or later recommended)
- Apple Silicon Mac (M1 or later)
- Python 3.10+

## Time & risk

- **Duration:** 10 minutes
- **Risk level:** Low — pip install
- **Rollback:** `pip uninstall mlx-lm`

<!-- tab: Install -->
## Install

```bash
pip install mlx-lm
```

## Run CLI inference

```bash
mlx_lm.generate \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --prompt "Explain the unified memory architecture of Apple Silicon." \
  --max-tokens 512
```

For 32B models (requires 32GB+ RAM):
```bash
mlx_lm.generate \
  --model mlx-community/Qwen2.5-32B-Instruct-4bit \
  --prompt "Hello!" \
  --max-tokens 256
```

## Start an OpenAI-compatible server

```bash
mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8080
```

<!-- tab: Examples -->
## Python API

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")
response = generate(
    model, tokenizer,
    prompt="What is Metal GPU acceleration?",
    max_tokens=256,
    verbose=True  # prints tokens/sec
)
print(response)
```

## Chat with system prompt

```python
from mlx_lm import load, generate
from mlx_lm.utils import get_model_path

model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What are the best Mac models for ML?"},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
response = generate(model, tokenizer, prompt=prompt, max_tokens=512, verbose=True)
print(response)
```

<!-- tab: Troubleshooting -->
## Out of memory

Use a more aggressively quantized model (3-bit or 2-bit):
```bash
mlx_lm.generate --model mlx-community/Qwen2.5-7B-Instruct-3bit --prompt "Hello!"
```

## Slow generation

Check no other GPU-intensive processes are running. Use Activity Monitor → GPU History.

## M5 compatibility

For M5 with Neural Accelerators: ensure macOS 26.2+ and latest MLX:
```bash
pip install -U mlx mlx-lm
```
