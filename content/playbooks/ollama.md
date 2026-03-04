---
slug: ollama
title: Ollama
time: "5 min"
color: green
desc: "Install and run LLMs locally with a single command"
tags: [inference]
spark: "Ollama"
category: onboarding
featured: true
whatsNew: false
---

<!-- tab: Overview -->
## Basic idea

Ollama is the simplest path to running LLMs on your Mac. One install, one command, and you're chatting with a model.

It wraps llama.cpp with a clean CLI and REST API, managing model downloads, quantization, and serving automatically.

## What you'll accomplish

You will have Ollama running on your Mac with models accessible via CLI and API at localhost:11434.

## Prerequisites

- macOS 13.0+ (Ventura or later)
- Apple Silicon Mac (M1 or later)
- 8 GB+ unified memory (16 GB+ recommended)

## Time & risk

- **Duration:** 5 minutes
- **Risk level:** None — standard app install
- **Rollback:** `brew uninstall ollama`

<!-- tab: Install -->
## Step 1: Install Ollama

```bash
brew install ollama
```

Or download from https://ollama.com/download/mac

## Step 2: Start the server

```bash
ollama serve
```

## Step 3: Pull and run a model

```bash
ollama pull qwen2.5:7b
ollama run qwen2.5:7b
```

For machines with 32GB+:

```bash
ollama pull qwen2.5:32b
ollama run qwen2.5:32b
```

<!-- tab: API Usage -->
## Test the API

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "qwen2.5:7b",
  "messages": [{"role": "user", "content": "Write me a haiku about Apple Silicon."}],
  "stream": false
}'
```

## Memory Guidelines

| Unified Memory | Recommended Max Model Size |
|---|---|
| 8 GB | 7B Q4 |
| 16 GB | 14B Q4 or 7B FP16 |
| 32 GB | 32B Q4 or 14B FP16 |
| 64 GB | 70B Q4 or 32B FP16 |
| 96–192 GB | 70B FP16, 120B Q4+ |

<!-- tab: Troubleshooting -->
## Ollama is slow

Ensure no other large processes are consuming memory. Check Activity Monitor → Memory.

## Out of memory

Use `OLLAMA_MAX_LOADED_MODELS=1` to avoid loading multiple models.

Use a smaller quantized model.

## Server won't start

Check if another instance is already running:
```bash
pkill ollama
ollama serve
```
