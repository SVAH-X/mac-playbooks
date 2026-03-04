---
slug: llama-cpp
title: "llama.cpp with Metal"
time: "15 min"
color: green
desc: "Run GGUF models with first-class Apple Silicon Metal acceleration"
tags: [inference, metal]
spark: "TRT-LLM / Nemotron with llama.cpp"
category: inference
featured: false
whatsNew: false
---

<!-- tab: Overview -->
## Basic idea

llama.cpp treats Apple Silicon as a first-class citizen with optimized Metal kernels, ARM NEON, and Accelerate framework integration. It's the runtime behind Ollama and LM Studio, but you can use it directly for maximum control.

## Prerequisites

- macOS 13.0+
- Apple Silicon Mac (M1 or later)
- Xcode Command Line Tools
- A GGUF model file

## Time & risk

- **Duration:** 15 minutes (longer if building from source)
- **Risk level:** Low

<!-- tab: Install -->
## Option 1: Homebrew

```bash
brew install llama.cpp
```

## Option 2: Build from source (recommended for latest optimizations)

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j$(sysctl -n hw.ncpu)
```

## Download a model

Models in GGUF format are available on Hugging Face. Example:
```bash
# Using huggingface-cli
pip install huggingface-hub
huggingface-cli download bartowski/Qwen2.5-7B-Instruct-GGUF \
  --include "Qwen2.5-7B-Instruct-Q4_K_M.gguf" \
  --local-dir ~/models
```

<!-- tab: Run -->
## CLI inference

```bash
llama-cli \
  -m ~/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  -ngl 99 \
  -c 8192 \
  -p "Explain how Metal acceleration works for LLM inference."
```

## Start an OpenAI-compatible server

```bash
llama-server \
  -m ~/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  -ngl 99 \
  -c 8192 \
  --port 8080
```

## Key flags

- `-ngl 99` — offload all layers to Metal GPU (critical for performance)
- `-c 8192` — context window size (adjust based on memory)
- `-t $(sysctl -n hw.perflevel0.logicalcpu)` — use only performance cores

<!-- tab: Troubleshooting -->
## Metal not being used

Look for `ggml_metal_init` in the output. If missing, rebuild with `-DGGML_METAL=ON`.

## Build fails

Install Xcode command line tools:
```bash
xcode-select --install
```

## Out of memory

Reduce context size with a smaller `-c` value, or use a more quantized model (Q2_K instead of Q4_K_M).
