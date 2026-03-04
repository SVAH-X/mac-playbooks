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

MLX is Apple's machine learning framework built from scratch for the unified memory architecture of Apple Silicon. Unlike PyTorch (which was designed for CUDA and then ported to Apple's MPS backend) or llama.cpp (written for CPU first and then accelerated with Metal), MLX was designed from the ground up assuming that CPU and GPU share a single memory pool with zero latency between them.

The practical consequence: MLX never copies tensors between CPU and GPU memory. On a discrete GPU system (NVIDIA), loading a model means copying weights from system RAM to VRAM — this takes seconds and consumes VRAM capacity separately from system RAM. On Apple Silicon with MLX, weights live in unified memory and are accessible to both CPU and GPU simultaneously. There is no copy step and no VRAM cap separate from your total RAM.

MLX LM is the language model layer on top of MLX. It provides the model loading, tokenization, and sampling logic that turns the low-level MLX framework into something you can point at a Hugging Face model ID and get text out of.

Ollama also uses Apple Silicon acceleration (via llama.cpp's Metal backend), but MLX LM consistently benchmarks 10–30% faster on the same models because MLX's kernels are tuned specifically for the Apple Silicon memory hierarchy rather than being adapted from CUDA-first code.

## What you'll accomplish

After following this playbook you will have:

- CLI inference at the highest available tokens/sec on your Mac (`mlx_lm.generate`)
- A running OpenAI-compatible API server you can hit with `curl` or any OpenAI SDK
- Python API access to run inference programmatically from your own code
- The ability to download and run any model from the `mlx-community` org on Hugging Face

## What to know before starting

- **What Hugging Face model hub is:** A public repository of pre-trained model weights, tokenizers, and configs. When you run `mlx_lm.generate --model mlx-community/Qwen2.5-7B-Instruct-4bit`, MLX LM downloads the model from Hugging Face and caches it in `~/.cache/huggingface/hub/`. You need an internet connection for the first download; after that, inference is fully offline.

- **What the mlx-community org is:** A Hugging Face organization that maintains pre-converted MLX versions of popular models. Original models are released in PyTorch format (safetensors). mlx-community converts them to MLX-compatible quantized format and publishes them. You don't need to convert models yourself unless you want a specific quantization that doesn't exist yet.

- **What safetensors format is:** The file format MLX uses for model weights. It's a tensor serialization format that is faster to load than PyTorch's pickle-based `.bin` format and doesn't have security issues with untrusted weights. MLX model repos on Hugging Face contain `.safetensors` files alongside the tokenizer configs.

- **What tokenization is:** LLMs operate on "tokens," not characters. A tokenizer converts input text into a sequence of integer IDs (e.g., "Apple Silicon" → [15789, 22153]) and converts output IDs back to text. MLX LM loads the correct tokenizer for each model automatically. This matters because different models have different vocabularies and different chat formatting conventions.

- **Lazy vs. eager evaluation:** MLX uses lazy evaluation. When you write `y = mlx.core.matmul(a, b)`, nothing is computed yet — MLX records the operation in a computation graph. Computation only happens when the result is needed (e.g., when you call `.tolist()` or when `generate()` needs the next token probability). This allows MLX to optimize the full computation graph before executing, and it is why the first generation call is slightly slower than subsequent ones.

## Prerequisites

- macOS 14.0+ Sonoma (macOS 15 Sequoia recommended; MLX gets performance improvements with each OS release)
- Apple Silicon (M1 or later) — MLX does not run on Intel Macs
- Python 3.10, 3.11, or 3.12 (3.13 support depends on MLX release)
- `pip` or a virtual environment manager (`uv`, `conda`, `venv`)

## Time & risk

- **Duration:** 10 minutes (plus model download time — a 7B 4-bit model is ~4.3 GB)
- **Risk level:** Low — pip install into user Python; no system changes
- **Rollback:** `pip uninstall mlx-lm mlx`

<!-- tab: Install -->
## Step 1: Install mlx-lm

`pip install mlx-lm` installs two packages: `mlx` (the core framework from Apple) and `mlx-lm` (the language model layer). It also pulls in `huggingface-hub` for model downloading and `transformers` for tokenizer support.

```bash
# Install into your current Python environment
pip install mlx-lm

# Verify the install worked
python -c "import mlx_lm; print('mlx-lm installed successfully')"

# Check the version
pip show mlx-lm | grep Version
```

If you use a virtual environment (recommended to avoid dependency conflicts):

```bash
python3 -m venv ~/.venvs/mlx
source ~/.venvs/mlx/bin/activate
pip install mlx-lm
```

To get the latest MLX improvements (MLX releases frequently with performance gains):

```bash
pip install -U mlx mlx-lm
```

## Step 2: Run your first inference with the CLI

`mlx_lm.generate` is the CLI entry point. The `--model` flag accepts a Hugging Face repo ID. On first run, it downloads the model weights to `~/.cache/huggingface/hub/` and then runs inference. Subsequent runs load from cache and start in seconds.

```bash
mlx_lm.generate \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \   # HuggingFace model ID (downloads on first use)
  --prompt "Explain the unified memory architecture of Apple Silicon in two paragraphs." \
  --max-tokens 512 \                                  # stop generating after 512 tokens
  --verbose                                           # print tokens/sec and memory usage after generation
```

Expected output (abbreviated):

```
==========
Explain the unified memory architecture of Apple Silicon...
[model response streams here]
==========
Prompt: 12 tokens, 45.2 tokens-per-second
Generation: 512 tokens, 48.7 tokens-per-second
Peak memory: 4.82 GB
```

The `--verbose` flag is extremely useful: it shows you actual throughput and peak memory usage, which lets you compare models and find the best fit for your machine.

Model sizes available in mlx-community (all 4-bit quantized):

| Model | HF repo | RAM at 4-bit | Speed on M2 Pro |
|---|---|---|---|
| Qwen2.5 3B | `mlx-community/Qwen2.5-3B-Instruct-4bit` | ~2 GB | ~90 tok/s |
| Qwen2.5 7B | `mlx-community/Qwen2.5-7B-Instruct-4bit` | ~4.3 GB | ~50 tok/s |
| Qwen2.5 14B | `mlx-community/Qwen2.5-14B-Instruct-4bit` | ~8.5 GB | ~28 tok/s |
| Qwen2.5 32B | `mlx-community/Qwen2.5-32B-Instruct-4bit` | ~20 GB | ~14 tok/s |

## Step 3: Start the OpenAI-compatible server

`mlx_lm.server` starts an HTTP server that accepts requests in OpenAI's API format. This means any code that works with OpenAI's Python SDK or API can be pointed at this local server instead.

```bash
mlx_lm.server \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \   # model to serve
  --port 8080 \                                       # port to listen on (default: 8080)
  --max-tokens 2048                                   # maximum context + generation length
```

Expected output:

```
Loading model from mlx-community/Qwen2.5-7B-Instruct-4bit...
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8080
```

Test the server is working:

```bash
# List available models
curl http://localhost:8080/v1/models

# Send a chat completion request
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Hello, what can you do?"}],
    "max_tokens": 100
  }'
```

Available endpoints:
- `POST /v1/chat/completions` — OpenAI-compatible chat
- `POST /v1/completions` — raw text completion
- `GET /v1/models` — list loaded models

## Step 4: Use the Python API directly

The Python API gives you fine-grained control over generation parameters and lets you integrate MLX LM into larger applications. `load()` returns the model and tokenizer; `generate()` runs inference.

```python
from mlx_lm import load, generate

# Load model and tokenizer from Hugging Face (cached after first download)
# load() is the expensive call — do it once and reuse the objects
model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")

# Simple generation
response = generate(
    model,
    tokenizer,
    prompt="What is the unified memory advantage of Apple Silicon?",
    max_tokens=256,
    verbose=True,   # prints tokens/sec to stderr
)
print(response)
```

For chat-formatted prompts (models like Qwen, Llama-Instruct expect a specific format):

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")

# Build a chat-formatted prompt using the model's own chat template
messages = [
    {"role": "system", "content": "You are a helpful assistant. Be concise."},
    {"role": "user", "content": "What are the best Mac models for ML inference in 2025?"},
]
# apply_chat_template converts the messages list into the exact string format
# the model was fine-tuned on (e.g., <|im_start|>user ... <|im_end|>)
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

response = generate(model, tokenizer, prompt=prompt, max_tokens=512, verbose=True)
print(response)
```

<!-- tab: Examples -->
## Streaming generation

For long outputs, use the `stream_generate` function to process tokens as they arrive rather than waiting for the full response:

```python
from mlx_lm import load, stream_generate

model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")

messages = [{"role": "user", "content": "Write a detailed explanation of quantization."}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# stream_generate is a generator that yields one token at a time
for token in stream_generate(model, tokenizer, prompt=prompt, max_tokens=1024):
    print(token, end="", flush=True)
print()  # newline after streaming finishes
```

## Using with the OpenAI Python SDK

Point the OpenAI SDK at your local `mlx_lm.server` instance:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",   # mlx_lm.server doesn't check auth but SDK requires a value
)

response = client.chat.completions.create(
    model="mlx-community/Qwen2.5-7B-Instruct-4bit",
    messages=[
        {"role": "system", "content": "You are a Python expert."},
        {"role": "user", "content": "Explain Python's GIL in simple terms."},
    ],
    max_tokens=256,
)
print(response.choices[0].message.content)
```

## Performance comparison

Benchmarks on M2 Pro 16 GB, `qwen2.5:7b` at 4-bit quantization, 200-token output:

| Runtime | Tokens/sec | Notes |
|---|---|---|
| MLX LM | ~50 tok/s | Native Apple Silicon framework |
| Ollama (llama.cpp Metal) | ~42 tok/s | Good default choice, easiest setup |
| llama.cpp direct (Metal) | ~40 tok/s | Same backend as Ollama |
| PyTorch MPS | ~18 tok/s | MPS backend is not as optimized as MLX |

MLX's advantage grows on larger models and longer context lengths where the unified memory architecture has more opportunity to avoid copies.

## Performance optimization tips

```bash
# Use 8-bit quantization for better quality at similar speed
mlx_lm.generate \
  --model mlx-community/Qwen2.5-7B-Instruct-8bit \
  --prompt "Hello" \
  --verbose

# Use 2-bit quantization to fit larger models in less RAM (quality tradeoff)
mlx_lm.generate \
  --model mlx-community/Qwen2.5-14B-Instruct-2bit \
  --prompt "Hello" \
  --verbose

# Keep the model loaded between calls using the server instead of CLI
# (CLI loads and unloads the model for each call, server keeps it in memory)
mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit &
curl http://localhost:8080/v1/chat/completions -d '...'
```

<!-- tab: Troubleshooting -->
## Quick reference

| Symptom | Likely cause | Fix |
|---|---|---|
| `OSError: model not found` on first run | Hugging Face download failed (auth or network) | Check internet connection; for gated models, run `huggingface-cli login` |
| Process killed during generation | OOM — model too large for available RAM | Use a more quantized variant (4-bit or 2-bit) or a smaller model |
| `ValueError: Tokenizer class not found` | Model uses a custom tokenizer not in transformers | `pip install -U transformers` |
| First generation call takes 30+ seconds | MLX is JIT-compiling the graph on first call | Normal behavior — subsequent calls are fast |
| Server returns 500 on /v1/chat/completions | Wrong message format or missing `add_generation_prompt` | Ensure messages list has correct role values: "system", "user", "assistant" |
| `mlx_lm: command not found` | mlx-lm installed in a different Python env | Activate the correct venv: `source ~/.venvs/mlx/bin/activate` |
| `mps not available` error | Wrong framework — not an MLX error | This is a PyTorch error; ensure you're using mlx_lm, not torch |
| Port 8080 already in use | Another server on that port | Use `--port 9090` or kill the conflicting process with `lsof -i :8080` |

## Model download fails

MLX LM downloads from Hugging Face. If the download fails or stalls:

```bash
# Check if you have the huggingface-hub CLI
pip install huggingface-hub

# Try downloading manually (shows clearer error messages)
huggingface-cli download mlx-community/Qwen2.5-7B-Instruct-4bit

# For gated models (Llama 3, Gemma, etc.) — requires HF account and agreement to model terms
huggingface-cli login   # prompts for your HF token
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct

# If download is corrupted, clear the cache and retry
rm -rf ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-7B-Instruct-4bit
mlx_lm.generate --model mlx-community/Qwen2.5-7B-Instruct-4bit --prompt "test"
```

## Out of memory

MLX uses unified memory, so OOM means your Mac is swapping to disk (very slow) or the process is killed by macOS memory pressure. Signs: generation slows dramatically then stops, or Activity Monitor shows memory pressure in the red.

```bash
# Check current memory usage
python3 -c "import mlx.core as mx; print(mx.metal.device_info())"
# Look for: "memory_size" and current allocation

# Solutions in order of preference:
# 1. Use a lower bit-width quantization
mlx_lm.generate --model mlx-community/Qwen2.5-7B-Instruct-3bit --prompt "test"

# 2. Use a smaller model
mlx_lm.generate --model mlx-community/Qwen2.5-3B-Instruct-4bit --prompt "test"

# 3. Reduce max_tokens to limit KV cache growth
mlx_lm.generate --model mlx-community/Qwen2.5-7B-Instruct-4bit --prompt "test" --max-tokens 256

# 4. Free up RAM by closing Chrome, other memory-heavy apps
```

## Slow generation on first call (JIT compilation)

MLX compiles computation graphs lazily. The very first inference call triggers compilation, which can take 10–30 seconds. This is normal and expected. Every call after the first uses the compiled graph and runs at full speed.

To warm up the model before your actual workload:

```python
model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")

# Warmup call — runs JIT compilation, result is discarded
_ = generate(model, tokenizer, prompt="Hello", max_tokens=1, verbose=False)

# Now your actual calls will be at full speed
response = generate(model, tokenizer, prompt="Your real prompt", max_tokens=512, verbose=True)
```

## Updating MLX for new hardware support

Apple releases MLX updates that add support for new chip features (including new M-series chips). If you're on a recently released Mac or macOS version and seeing errors or unexpectedly low performance:

```bash
# Update both mlx and mlx-lm to the latest versions
pip install -U mlx mlx-lm

# Verify the new version
python -c "import mlx; print(mlx.__version__)"
```