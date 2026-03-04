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

Ollama packages llama.cpp with an automatic model manager, a REST API server, and a CLI into a single installable binary. You do not manage model files, choose quantization formats, or configure GPU settings manually — Ollama handles all of that.

Under the hood, Ollama uses llama.cpp's Metal backend to dispatch matrix multiplications to your Mac's GPU. On NVIDIA hardware (Linux/Windows), the same workload runs through CUDA. On Apple Silicon, it goes through Metal — Apple's GPU compute framework — because Apple Silicon has no CUDA support. Ollama abstracts this entirely so you interact with the same CLI and API regardless of hardware.

The practical benefit: you run `ollama run qwen2.5:7b` and get a working chatbot. You don't need to know what a GGUF file is, how many GPU layers to offload, or what quantization format to choose.

## What you'll accomplish

After following this playbook you will have:

- Ollama running as a background service on port 11434
- `qwen2.5:7b` downloaded and cached locally (~4.7 GB on disk)
- A working interactive CLI chat session with the model
- A working `curl` call to the REST API confirming the model responds
- A rough tokens/sec baseline from the built-in benchmark

On an M2 Pro with 16 GB RAM, `qwen2.5:7b` at Q4_K_M quantization runs at roughly 40–55 tokens/sec.

## What to know before starting

- **What LLMs are:** Large language models are next-token predictors. Given a sequence of text tokens, they predict the probability distribution of the next token and sample from it. "Generating text" is just doing this thousands of times in sequence. They are not databases and do not look things up — they predict plausible continuations based on patterns learned during training.

- **What quantization means:** A 7B-parameter model in full 32-bit float precision requires ~28 GB of RAM. Quantization reduces each weight from 32-bit to fewer bits (4-bit in Q4_K_M). The 7B model then fits in ~4.7 GB. Quality degrades slightly but is usually imperceptible for chat tasks. Ollama downloads Q4_K_M by default for most models.

- **What an API server means:** `ollama serve` starts an HTTP server. Clients send JSON requests describing the model and messages; the server runs inference and returns JSON responses. This lets any app — Python scripts, web UIs, IDEs — use local models without embedding the inference engine themselves.

- **What Metal is:** Metal is Apple's GPU compute and graphics framework. When llama.cpp (inside Ollama) runs a matrix multiplication, it dispatches it as a Metal shader to the GPU cores in your M-series chip. This is what makes inference fast — without Metal, every matrix operation would run on the CPU only.

- **Why unified memory matters:** On Apple Silicon, CPU and GPU share the same physical RAM pool. There is no separate "VRAM." A 16 GB M2 has 16 GB accessible to both CPU and GPU simultaneously. This means a 7B Q4 model loaded into RAM is already in GPU-accessible memory — no copying required. On discrete GPUs (NVIDIA), models must be copied from system RAM to VRAM before inference can begin.

## Prerequisites

- macOS 13.0+ Ventura (macOS 14 Sonoma recommended)
- Apple Silicon Mac (M1 or later) — Intel Macs are supported but Metal GPU acceleration is Apple Silicon only
- Homebrew installed (`/opt/homebrew/bin/brew` present) OR ability to download a .pkg installer
- 8 GB+ unified memory (16 GB+ recommended for 7B models to leave headroom for other apps)

## Time & risk

- **Duration:** 5 minutes
- **Risk level:** None — standard Homebrew formula, no system modifications
- **Rollback:** `brew uninstall ollama && rm -rf ~/.ollama`

<!-- tab: Install -->
## Step 1: Install Ollama via Homebrew

Homebrew has a first-party Ollama formula. No tap is needed — it is in the core Homebrew repository. The formula installs the `ollama` binary to `/opt/homebrew/bin/ollama` and sets up the necessary entitlements for Metal access.

```bash
brew install ollama
# Expected output: ==> Installing ollama
# This downloads ~50 MB binary and takes about 30 seconds on a good connection.
```

If you prefer not to use Homebrew, download the `.pkg` installer directly from https://ollama.com/download/mac. The .pkg also installs a menu bar app that manages the server process automatically. The Homebrew path gives you more control (no menu bar app, easier to script).

Verify the install:

```bash
ollama --version
# Expected: ollama version 0.x.x
```

## Step 2: Start the Ollama server

`ollama serve` starts an HTTP server that listens on `0.0.0.0:11434` by default. It runs in the foreground — you will see log output as requests come in. For development and exploration, running it in the foreground is fine because you can see errors immediately.

```bash
ollama serve
# Expected output:
# time=2024-01-01T00:00:00.000Z level=INFO source=routes.go msg="Listening on [::]:11434"
# Leave this terminal open. Open a new terminal tab for the next steps.
```

To run it in the background instead:

```bash
# Option A: background with & (does not survive terminal close)
ollama serve &

# Option B: start as a launchd service (survives reboot, recommended for daily use)
brew services start ollama
# Verify: brew services list | grep ollama
```

To change the port or bind address, set the environment variable before starting:

```bash
OLLAMA_HOST=127.0.0.1:11435 ollama serve   # different port, localhost only
```

## Step 3: Pull your first model

Model names in Ollama follow the format `name:size-variant`. The tag after `:` specifies the parameter count and quantization. If you omit the tag, Ollama picks a sensible default (usually Q4_K_M for the most popular size).

```bash
# Pull Qwen 2.5 7B — a strong general-purpose model that fits in 8 GB RAM
ollama pull qwen2.5:7b
# Expected: pulls layers progressively, shows download progress
# Total download: ~4.7 GB
# Stored in: ~/.ollama/models/
```

Memory requirements by model size (all Q4_K_M quantization):

| Model size | RAM needed | Recommended machine |
|---|---|---|
| 1B | ~1 GB | Any Mac |
| 3B | ~2 GB | Any Mac |
| 7B | ~5 GB | 8 GB Mac (leaves ~3 GB for OS) |
| 14B | ~9 GB | 16 GB Mac |
| 32B | ~20 GB | 32 GB Mac |
| 70B | ~43 GB | 64 GB Mac |

For machines with less RAM, try a 3B model: `ollama pull qwen2.5:3b`

## Step 4: Run an interactive chat session

`ollama run` starts an interactive REPL. You type messages, press Enter, and see the model's response stream in real time. Under the hood this is hitting the local REST API.

```bash
ollama run qwen2.5:7b
# Starts the interactive chat. You should see a >>> prompt.

# Inside the chat:
# >>> Hello! What can you tell me about Apple Silicon?
# (model streams response)
# >>> /list     — shows all loaded models
# >>> /bye      — exits the chat
```

To send a single non-interactive prompt (useful for scripting):

```bash
ollama run qwen2.5:7b "Explain quantization in one paragraph."
# Prints the response and exits immediately
```

Run a quick benchmark to see your tokens/sec:

```bash
ollama run qwen2.5:7b "" --verbose
# The --verbose flag prints tokens/sec, load time, and eval time after generation
```

<!-- tab: API Usage -->
## The REST API

Ollama's REST API is the interface that tools like Open WebUI, Continue.dev, and Cursor use to talk to your local models. Understanding it lets you build your own integrations.

The base URL is always `http://localhost:11434`. There are two API styles: the native Ollama API and the OpenAI-compatible API. Use the OpenAI-compatible one when you want to swap in Ollama as a drop-in replacement for OpenAI's API.

## Native API: /api/chat

The `/api/chat` endpoint accepts a conversation as an array of messages and returns a response. Set `"stream": false` to get the full response in one JSON blob; omit it or set `true` for streaming (each token arrives as a separate JSON line).

```bash
# Non-streaming: get the full response as one JSON object
curl http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:7b",
    "messages": [
      {"role": "user", "content": "Write a haiku about Apple Silicon."}
    ],
    "stream": false
  }'
# Expected: {"model":"qwen2.5:7b","message":{"role":"assistant","content":"..."},...}

# Streaming: each line is a JSON object with a partial token
curl http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:7b",
    "messages": [{"role": "user", "content": "Count to 5."}],
    "stream": true
  }'
```

## Native API: /api/tags

List all downloaded models:

```bash
curl http://localhost:11434/api/tags
# Returns JSON with all local models, their sizes, and modification dates
```

## OpenAI-compatible API: /v1/chat/completions

Ollama exposes an OpenAI-compatible endpoint. Any code written for OpenAI's API works with Ollama by changing just the `base_url` and `model` name.

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ollama" \
  -d '{
    "model": "qwen2.5:7b",
    "messages": [
      {"role": "system", "content": "You are a concise assistant."},
      {"role": "user", "content": "What is Metal GPU acceleration?"}
    ]
  }'
```

## Python with the openai SDK

```python
from openai import OpenAI

# Point the OpenAI SDK at your local Ollama server
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Ollama ignores the key but the SDK requires a non-empty value
)

response = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What makes Apple Silicon fast for ML inference?"},
    ],
)
print(response.choices[0].message.content)
```

## Memory guidelines

| Unified Memory | Recommended max model |
|---|---|
| 8 GB | 7B Q4 |
| 16 GB | 14B Q4 or 7B FP16 |
| 32 GB | 32B Q4 or 14B FP16 |
| 64 GB | 70B Q4 or 32B FP16 |
| 96–192 GB | 70B FP16, 120B Q4+ |

<!-- tab: Troubleshooting -->
## Quick reference

| Symptom | Likely cause | Fix |
|---|---|---|
| `curl: (7) Failed to connect` | Server not running | Run `ollama serve` in a separate terminal |
| `model "qwen2.5:7b" not found` | Model not pulled yet | Run `ollama pull qwen2.5:7b` |
| Process killed / app crashes | Not enough RAM for model | Use a smaller model or `OLLAMA_MAX_LOADED_MODELS=1` |
| Very slow generation (<5 tok/s) | GPU not being used, or too many layers on CPU | Check GPU usage; see "Slow generation" below |
| `bind: address already in use` | Another Ollama instance running | Run `pkill ollama` then `ollama serve` |
| `failed to load model` | Corrupted download or wrong quantization | Delete model file and re-pull |
| Disk full errors | Models dir too large | Run `ollama list` and `ollama rm <model>` to remove unused models |

## Out of memory (OOM) crashes

When the model plus its KV cache (used to store context during generation) exceeds your available RAM, macOS will kill the process. This shows up as the terminal suddenly returning to a prompt, or the server process disappearing.

How to diagnose: install `asitop` (`pip install asitop`) and run it in another terminal while doing inference. It shows GPU memory usage in real time. If you see the number climbing to your RAM ceiling, you are close to OOM.

Fixes in order of preference:

```bash
# Option 1: Use a more aggressively quantized version of the same model
ollama pull qwen2.5:7b-instruct-q2_k   # smaller but lower quality

# Option 2: Reduce context size (KV cache grows with context length)
OLLAMA_NUM_CTX=2048 ollama run qwen2.5:7b

# Option 3: Prevent Ollama from keeping multiple models loaded at once
OLLAMA_MAX_LOADED_MODELS=1 ollama serve

# Option 4: Switch to a smaller model
ollama pull qwen2.5:3b
```

## Slow generation (low tokens/sec)

If you are getting fewer than 10 tokens/sec on a 7B model, the GPU is likely not being used or is partially being used.

```bash
# Check if ollama is seeing the GPU
ollama run qwen2.5:7b "" --verbose
# Look for: "num_gpu_layers: 33" — all layers on GPU
# If you see "num_gpu_layers: 0" — nothing is on GPU

# Force GPU layers (via env var before starting server)
OLLAMA_NUM_GPU_LAYERS=99 ollama serve
```

Also check that no other GPU-intensive process is running (browser hardware acceleration, video processing, another model server).

## Port already in use

If something else is on port 11434, find and kill it:

```bash
# Find what's using port 11434
lsof -i :11434

# Kill all ollama processes
pkill ollama

# Start on a different port
OLLAMA_HOST=0.0.0.0:11435 ollama serve
```

## Models directory too large

Ollama stores all models in `~/.ollama/models/`. Each model is 2–40 GB depending on size and quantization.

```bash
# List all downloaded models with sizes
ollama list

# Remove a model you no longer need
ollama rm qwen2.5:7b

# Check total disk usage
du -sh ~/.ollama/models/
```