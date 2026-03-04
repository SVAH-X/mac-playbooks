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

llama.cpp is the foundational C++ implementation of LLM inference that powers Ollama, LM Studio, and many other popular tools. When you run `ollama run qwen2.5:7b`, Ollama is invoking llama.cpp's Metal backend under the hood. Running llama.cpp directly removes that abstraction layer and gives you fine-grained control over every aspect of inference.

Why use llama.cpp directly instead of Ollama? Three reasons:

1. **Control:** You choose exactly how many GPU layers to offload (`-ngl`), what context size to allocate (`-c`), how many threads to use, and what sampling strategy to apply. Ollama makes sensible defaults; llama.cpp lets you tune for your specific hardware and use case.

2. **GGUF format:** llama.cpp uses GGUF — a single-file model format that packages weights and quantization metadata together. You can download individual GGUF files from Hugging Face and run them directly without a model manager. This is useful when you need a specific quantization variant that Ollama doesn't expose.

3. **Transparency:** You see exactly what the runtime is doing. The verbose output shows Metal initialization, layer allocation, memory usage, and tokens/sec all in your terminal.

On NVIDIA hardware, you'd use TensorRT-LLM or vLLM for maximum performance. On Apple Silicon, llama.cpp with Metal is the C++ equivalent — highly optimized for the hardware, maximum control.

## What you'll accomplish

After following this playbook you will have:

- `llama-cli` installed and running interactive chat sessions via the command line
- `llama-server` running an OpenAI-compatible HTTP API on port 8080
- A downloaded GGUF model file running with full Metal GPU acceleration
- An understanding of the key flags so you can tune performance for your machine

## What to know before starting

- **What GGUF format is:** GGUF (GPT-Generated Unified Format) is a single-file model format designed for llama.cpp. A GGUF file contains model weights, quantization metadata, tokenizer vocabulary, and model architecture information all in one file. You download one file and run it. Compare this to Hugging Face's multi-file safetensors format used by PyTorch and MLX.

- **What Metal GPU layers means:** A transformer model consists of multiple "layers" (a 7B model typically has 28–32 layers). Each layer is a set of matrix multiplications. The `-ngl N` flag tells llama.cpp to offload `N` layers to the Metal GPU. More layers on GPU = faster inference. If you set `-ngl 99` (more than the model has), all layers go to GPU. If you have less RAM, reduce this number to leave some layers on CPU and reduce peak memory usage.

- **What quantization variants mean:** GGUF files come in many quantization levels, each a tradeoff between quality and size:
  - `Q2_K` — 2-bit, aggressive compression, fits large models in small RAM, noticeable quality loss
  - `Q4_K_M` — 4-bit with K-means grouping, Mixed precision — the standard recommendation, good balance of quality and size
  - `Q5_K_M` — 5-bit, noticeably higher quality than Q4 at ~20% more RAM
  - `Q8_0` — 8-bit, near-lossless quality, uses ~2x the RAM of Q4
  - `F16` — full 16-bit float, maximum quality, uses ~2x the RAM of Q8

- **llama.cpp's lineage:** llama.cpp was created by Georgi Gerganov in January 2023 after the original LLaMA model weights leaked from Facebook/Meta. It was initially a pure CPU C++ implementation. Metal GPU support was added within weeks by the community. It now supports Llama, Qwen, Mistral, Phi, Gemma, and nearly every major open model architecture. Ollama, LM Studio, Jan, and many other tools use llama.cpp as their inference engine.

## Prerequisites

- macOS 12.0+ (macOS 14 Sonoma recommended for latest Metal optimizations)
- Apple Silicon (M1 or later) — Intel Macs work but without Metal acceleration
- Xcode Command Line Tools: run `xcode-select --install` if you haven't already
- Homebrew (for the Homebrew install path) OR CMake 3.14+ (for the from-source path)
- A GGUF model file — you will download one in the Install tab

## Time & risk

- **Duration:** 15 minutes via Homebrew, 25–30 minutes if building from source
- **Risk level:** Low — standard CLI tools, no system modifications
- **Rollback:** `brew uninstall llama.cpp` (Homebrew path) or `rm -rf ~/llama.cpp/build` (source path)

<!-- tab: Install -->
## Step 1: Install via Homebrew (recommended)

Homebrew maintains the `llama.cpp` formula, which installs the latest stable release of `llama-cli` (interactive chat) and `llama-server` (HTTP API server) to `/opt/homebrew/bin/`.

```bash
brew install llama.cpp
# Installs: llama-cli, llama-server, and supporting libraries
# Takes about 1-2 minutes on a fast connection

# Verify installation
llama-cli --version
# Expected: version b... (build number)

llama-server --version
# Expected: same version
```

Note on binary naming: the CLI tool is `llama-cli`, not `llama.cpp` or `llama`. The server is `llama-server`. Older builds used `main` and `server` — if you see tutorials using those names, they are outdated.

## Step 2 (alternative): Build from source

Building from source gives you the absolute latest commits including unreleased Metal optimizations, and lets you compile with custom flags for your specific chip. This is worth the extra time if you want maximum performance or need a feature not yet in the Homebrew release.

```bash
# Clone the repository
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# Configure the build with Metal enabled
# -DGGML_METAL=ON: enables Metal GPU acceleration (required for Apple Silicon)
# -DGGML_BLAS=ON: uses Apple's Accelerate framework for CPU BLAS operations
# -DCMAKE_BUILD_TYPE=Release: optimized build (much faster than Debug)
cmake -B build \
  -DGGML_METAL=ON \
  -DGGML_BLAS=ON \
  -DCMAKE_BUILD_TYPE=Release

# Build using all available CPU cores
# hw.ncpu returns the total logical core count (performance + efficiency cores)
cmake --build build --config Release -j$(sysctl -n hw.ncpu)
# Takes 5–10 minutes depending on your machine

# The binaries are in build/bin/
./build/bin/llama-cli --version

# Optional: add to PATH by symlinking
ln -s $(pwd)/build/bin/llama-cli /opt/homebrew/bin/llama-cli
ln -s $(pwd)/build/bin/llama-server /opt/homebrew/bin/llama-server
```

When to build from source vs Homebrew: if you are happy with the model quality and speed you see, use Homebrew. If you are benchmarking or chasing maximum tokens/sec, build from source and experiment with compiler flags.

## Step 3: Download a GGUF model

GGUF models are hosted on Hugging Face. The `huggingface-cli` tool downloads them efficiently with resume support. Model filenames encode everything you need to know about the model.

Reading a GGUF filename: `Qwen2.5-7B-Instruct-Q4_K_M.gguf`
- `Qwen2.5` — model architecture and family
- `7B` — 7 billion parameters
- `Instruct` — instruction-tuned variant (chat-optimized, not base model)
- `Q4_K_M` — 4-bit quantization with K-means grouping, Mixed precision
- `.gguf` — file format

```bash
# Install the Hugging Face CLI if you don't have it
pip install huggingface-hub

# Download Qwen2.5 7B Instruct Q4_K_M to ~/models/
# --include filters to just this file (the repo has many quantization variants)
# --local-dir sets where to save the file
huggingface-cli download bartowski/Qwen2.5-7B-Instruct-GGUF \
  --include "Qwen2.5-7B-Instruct-Q4_K_M.gguf" \
  --local-dir ~/models
# Downloads ~4.7 GB — takes 5–15 minutes depending on connection speed

# Verify the file exists and has the correct size
ls -lh ~/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf
# Expected: ~4.7G
```

Model file size reference (7B model at different quantization levels):

| Quantization | File size | RAM needed | Quality |
|---|---|---|---|
| Q2_K | ~2.7 GB | ~3.5 GB | Noticeably reduced |
| Q4_K_M | ~4.7 GB | ~6 GB | Recommended |
| Q5_K_M | ~5.4 GB | ~7 GB | Better than Q4 |
| Q8_0 | ~7.7 GB | ~10 GB | Near original quality |

Store all your GGUF models in `~/models/` — it's the de facto convention and makes paths easy to remember.

<!-- tab: Run -->
## Step 1: Basic CLI inference

`llama-cli` is the interactive chat and completion tool. The flags below are the most important ones — understanding them lets you tune for your specific model and hardware.

```bash
llama-cli \
  -m ~/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \   # path to GGUF model file
  -ngl 99 \                                          # offload all layers to Metal GPU
  -c 8192 \                                          # context window size in tokens
  -p "Explain how Metal GPU acceleration works for LLM inference on Apple Silicon." \
  -n 512 \                                           # generate up to 512 tokens
  -t $(sysctl -n hw.perflevel0.logicalcpu)           # use only performance cores (not efficiency cores)
```

Flag reference:

| Flag | Description | Recommendation |
|---|---|---|
| `-m` | Path to GGUF model file | Required |
| `-ngl N` | Offload N layers to Metal GPU | Set to 99 to offload all layers |
| `-c N` | Context window size in tokens | 8192 is a good default; reduce if OOM |
| `-p "..."` | Initial prompt text | Use for single-shot prompts |
| `-n N` | Max tokens to generate | 512 for short answers, 2048+ for long docs |
| `-t N` | Number of CPU threads | `sysctl -n hw.perflevel0.logicalcpu` for perf cores only |
| `--temp F` | Sampling temperature (0.0–2.0) | 0.0 = deterministic, 0.8 = default creative |
| `-i` | Interactive mode (chat REPL) | Use for back-and-forth conversations |
| `--chat-template` | Chat template name | Required for instruct models in interactive mode |

For interactive chat sessions with an instruct model:

```bash
llama-cli \
  -m ~/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  -ngl 99 \
  -c 8192 \
  -i \                        # interactive mode
  --chat-template qwen2       # use Qwen's chat template (required for correct formatting)
# Type your messages at the > prompt. Ctrl+C to exit.
```

Expected output when starting (confirms Metal is active):

```
ggml_metal_init: GPU name:   Apple M2 Pro
ggml_metal_init: GPU family: MTLGPUFamilyApple8
llm_load_tensors: offloading 28 repeating layers to GPU
llm_load_tensors: offloaded 28/28 layers to GPU
```

If you don't see `ggml_metal_init`, Metal is not working — see Troubleshooting.

## Step 2: Start the OpenAI-compatible server

`llama-server` runs an HTTP API server with an OpenAI-compatible `/v1/chat/completions` endpoint and a built-in web UI for testing at `http://localhost:8080`.

```bash
llama-server \
  -m ~/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  -ngl 99 \
  -c 8192 \
  --port 8080 \               # listen on port 8080
  --host 127.0.0.1            # bind to localhost only (more secure; use 0.0.0.0 to expose to network)
```

Test the server from another terminal:

```bash
# Chat completion request
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b",
    "messages": [
      {"role": "system", "content": "You are a concise assistant."},
      {"role": "user", "content": "What is GGUF format?"}
    ],
    "max_tokens": 200
  }'

# Check server health
curl http://localhost:8080/health
# Expected: {"status":"ok"}
```

The web UI is available at `http://localhost:8080` in your browser — useful for quick testing without writing any code.

## Performance by hardware

Benchmarks for `Qwen2.5-7B-Instruct-Q4_K_M.gguf` with `-ngl 99 -c 4096`:

| Mac | RAM | Tokens/sec (generation) |
|---|---|---|
| M1 8 GB | 8 GB | ~25–30 tok/s |
| M1 Pro 16 GB | 16 GB | ~35–40 tok/s |
| M2 Pro 16 GB | 16 GB | ~40–48 tok/s |
| M3 Pro 18 GB | 18 GB | ~48–55 tok/s |
| M4 Pro 24 GB | 24 GB | ~60–70 tok/s |

Generation speed is primarily GPU-bound when `-ngl 99` is set. If you see numbers significantly below these, check that Metal layers are being used (see Troubleshooting).

## Step 3: Speculative decoding

llama.cpp supports speculative decoding — using a small "draft" model to predict several tokens ahead, then verifying them with the main model. This can increase effective throughput by 2–3x on tasks with predictable output. See the speculative decoding playbook for a full walkthrough.

```bash
# Preview: speculative decoding with a draft model
llama-cli \
  -m ~/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \    # main model
  --model-draft ~/models/Qwen2.5-0.5B-Q4_K_M.gguf \ # small draft model
  -ngl 99 \
  --draft 8                                           # draft 8 tokens ahead
```

<!-- tab: Troubleshooting -->
## Quick reference

| Symptom | Likely cause | Fix |
|---|---|---|
| No `ggml_metal_init` in output | Metal not enabled in build | Reinstall via Homebrew or rebuild with `-DGGML_METAL=ON` |
| Build fails with CMake errors | Xcode CLT not installed or outdated | Run `xcode-select --install`; `sudo xcode-select --reset` |
| Very slow generation (<10 tok/s) | All layers on CPU, `-ngl` not set or set to 0 | Add `-ngl 99` to your command |
| `ggml_metal_init: failed to load default.metallib` | Missing Metal shader library | Reinstall llama.cpp or rebuild from source |
| Process killed during load | OOM — model too large for available RAM | Use lower quantization (Q2_K) or reduce `-c` to lower KV cache size |
| `context length exceeded` during generation | `-c` set too small | Increase `-c 16384`; reduce if that causes OOM |
| `error loading model` | Corrupted GGUF download | Delete the file and re-download; verify with `sha256sum` |
| `xcode-select: error` during build | Command Line Tools not installed | `xcode-select --install` |

## Metal not initializing

If you don't see `ggml_metal_init` in the llama-cli output, Metal GPU is not being used and you are running CPU-only — expect 3–5x slower performance.

Diagnose and fix:

```bash
# Check if Metal is available on your system
system_profiler SPDisplaysDataType | grep "Metal:"
# Expected: Metal: Supported, feature set Apple7 (or higher)

# If installed via Homebrew, reinstall to get the Metal-enabled build
brew uninstall llama.cpp
brew install llama.cpp

# If built from source, check the CMake configuration
cmake -B build -DGGML_METAL=ON   # ensure this flag is present

# Verify the build has Metal support
llama-cli --help | grep -i metal
# Should show Metal-related options
```

## Build fails

```bash
# Install or update Xcode Command Line Tools
xcode-select --install

# If tools are installed but outdated, reset the path
sudo xcode-select --reset

# Verify the C++ compiler is available
c++ --version
# Expected: Apple clang version 15.x or later

# If CMake is not found
brew install cmake

# Clean build directory and retry
rm -rf llama.cpp/build
cd llama.cpp
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(sysctl -n hw.ncpu)
```

## Slow inference (low tokens/sec)

The most common cause of slow inference is CPU-only execution because `-ngl` was not set or set to 0.

```bash
# Run with explicit debug output to see layer allocation
llama-cli \
  -m ~/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  -ngl 99 \
  -c 4096 \
  -p "hello" \
  -n 20 \
  --log-disable false   # enable verbose logging

# In output, look for:
# llm_load_tensors: offloading 28 repeating layers to GPU
# If it shows 0 layers offloaded, Metal is not working
```

Also check: another GPU-intensive process running (check Activity Monitor → GPU), or insufficient RAM causing heavy swap usage.

## Out of memory

```bash
# Check total model + KV cache memory requirements
# Rule of thumb: model file size + (context_size * num_layers * 2 * element_size)
# For 7B Q4_K_M with -c 8192: ~4.7 GB model + ~1.5 GB KV cache = ~6.2 GB

# Reduce context size to lower KV cache footprint
llama-cli -m ~/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf -ngl 99 -c 2048 -p "hello"

# Use a more aggressively quantized model
huggingface-cli download bartowski/Qwen2.5-7B-Instruct-GGUF \
  --include "Qwen2.5-7B-Instruct-Q2_K.gguf" \
  --local-dir ~/models

# Partially offload to GPU (if full offload causes OOM)
# Try progressively lower values
llama-cli -m ~/models/... -ngl 20   # offload first 20 layers only
llama-cli -m ~/models/... -ngl 10   # try even fewer
```