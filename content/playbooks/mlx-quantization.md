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

Quantization reduces model weights from 32-bit or 16-bit floating-point numbers to lower-precision integers (2–8 bit). This dramatically shrinks memory requirements: a 32B model at FP16 needs 64 GB of RAM — at 4-bit it needs only ~20 GB.

MLX uses **grouped quantization**: weights are divided into groups of 32 or 64 values, and each group gets its own learned scale factor and bias. This per-group scaling preserves more accuracy than naive per-tensor quantization, because local patterns in the weight matrix are captured individually. The quality loss at 4-bit is surprisingly small — perplexity increases by only 1–3%, which is imperceptible in conversational use.

During inference, the quantized weights are dequantized on the fly within the Metal shader — Apple Silicon's unified memory bandwidth handles this efficiently because the memory transfer is 4× smaller, and the dequantization math is simple multiply-add operations that run nearly for free on the GPU cores.

## What you'll accomplish

A locally quantized version of any Hugging Face model, ready to run with `mlx_lm.generate`. You'll also understand when to quantize yourself versus using pre-quantized models from the `mlx-community` organization, and how to evaluate quality degradation.

## What to know before starting

- **FP16/BF16** — 16-bit floating point formats use 2 bytes per parameter. A 7B model at FP16 = 7,000,000,000 × 2 bytes = 14 GB. BF16 (Brain Float) is similar but with a wider exponent range; both are common for inference.
- **INT4 / Q4** — 4-bit integer format uses 0.5 bytes per parameter. A 7B model at Q4 ≈ 3.5–4.5 GB (plus overhead for scales/biases). MLX's 4-bit format stores 2 weights per byte with per-group metadata.
- **Grouped quantization mechanics** — given a group of 64 weight values, find the min and max, then map each value to the nearest of 16 levels (for 4-bit). Store the scale (range/16) and zero-point alongside. At runtime, multiply stored integer by scale, add zero-point to recover approximately the original float.
- **Perplexity** — a measure of how surprised a language model is by a held-out text corpus. Lower perplexity = better language model. FP16 baseline for Qwen2.5-7B is ~8.2; at Q4_0 it rises to ~8.4 (+2.4%); at Q2 it might reach ~10.5 (+28%). The 4-bit increase is imperceptible in practice.
- **Why conversion needs 2× RAM** — the conversion process loads the full FP16 model (~14 GB for 7B), quantizes it layer by layer, then writes the quantized result (~4 GB). Both the full model and output must fit in memory simultaneously. After conversion, the original is no longer needed.

## Prerequisites

- macOS 14.0+ (Sonoma or later)
- Apple Silicon Mac
- Python 3.10, 3.11, or 3.12
- `mlx-lm` installed: `pip install mlx-lm`
- Hugging Face account (free); `huggingface-cli login` for gated models
- RAM: at least 2× the FP16 model size (7B needs 28+ GB free during conversion; 32B needs 128+ GB)

## Time & risk

- **Duration:** 10 minutes setup + conversion time (~5 min for 7B on M2 Max, ~20 min for 32B on M2 Max)
- **Risk level:** None — conversion creates new files and never modifies the original model
- **Rollback:** Delete the output directory; the source model on Hugging Face is unchanged

<!-- tab: Quantize -->
## Step 1: Log into Hugging Face

Many models require authentication to download. Even if you've used `huggingface-cli` before, confirm your token is valid — expired tokens cause silent 401 errors during conversion that look like network issues.

```bash
# Log in and store token in ~/.cache/huggingface/token
huggingface-cli login
# You'll be prompted for your token from https://huggingface.co/settings/tokens

# Verify the login worked
huggingface-cli whoami          # should print your username

# Check if a specific model is gated (requires agreement)
# Gated models show "Access to model X is restricted" on their HF page
# You must accept the license on the web before downloading
```

For non-gated models (most open models like Qwen, Mistral, Phi), you can skip login entirely. Gated models (Llama, Gemma) require both login and clicking "Agree" on the model's Hugging Face page.

## Step 2: Quantize to 4-bit

The `mlx_lm.convert` command downloads the source model, quantizes it layer by layer, and saves the result to a local directory. The output directory contains everything needed to run the model: config, tokenizer, and quantized weights.

```bash
mlx_lm.convert \
  --hf-path Qwen/Qwen2.5-7B-Instruct \    # source: HF repo name (org/model)
  --mlx-path ./qwen2.5-7b-4bit \           # destination: local directory to create
  --quantize \                             # enable quantization (without this, just converts format)
  --q-bits 4 \                             # 4 bits per weight — the standard choice
  --q-group-size 64                        # 64 weights per quantization group (default)
```

What gets created in `./qwen2.5-7b-4bit/`:

```
qwen2.5-7b-4bit/
├── config.json                    # model architecture configuration
├── tokenizer.json                 # tokenizer vocabulary and merge rules
├── tokenizer_config.json          # tokenizer settings including chat template
├── special_tokens_map.json        # special token IDs
├── model.safetensors              # quantized weights (or multiple shards for large models)
└── quantization.json              # per-layer quantization metadata (scales, biases, group size)
```

Conversion output to expect:
```
Fetching 9 files: 100%|████████| 9/9 [00:02<00:00]
Loading model from Qwen/Qwen2.5-7B-Instruct
Quantizing model to 4 bits
Saving to ./qwen2.5-7b-4bit
```

## Step 3: Run the quantized model to verify quality

Before deleting the source model or considering the conversion successful, test the output. Run a few prompts and compare the quality against what you'd expect from the model's reputation.

```bash
# Quick test — does it generate coherent output?
mlx_lm.generate \
  --model ./qwen2.5-7b-4bit \
  --prompt "What is the capital of France?" \
  --max-tokens 50

# Quality test — does it reason correctly?
mlx_lm.generate \
  --model ./qwen2.5-7b-4bit \
  --prompt "Explain the difference between a list and a tuple in Python." \
  --max-tokens 200

# Check memory usage during inference
# Activity Monitor > Memory tab > mlx_lm process should show ~4-5 GB for 7B Q4
```

If the output is garbled or obviously wrong, check the `--hf-path` value — a wrong path might have downloaded a different model than intended.

## Step 4: Upload to Hugging Face (optional)

If you want to share your quantized model with others or access it from multiple machines, upload it to Hugging Face.

```bash
# Upload the quantized model to your HF account
# Replace 'your-username' with your HF username
mlx_lm.convert \
  --hf-path Qwen/Qwen2.5-7B-Instruct \
  --mlx-path ./qwen2.5-7b-4bit \
  --quantize \
  --q-bits 4 \
  --upload-repo your-username/Qwen2.5-7B-Instruct-4bit-mlx

# Or upload an already-converted local model
huggingface-cli upload your-username/Qwen2.5-7B-Instruct-4bit-mlx ./qwen2.5-7b-4bit
```

<!-- tab: Guidelines -->
## Quantization bits comparison

| Bits | Size (7B model) | Perplexity increase | Quality assessment | Best for |
|---|---|---|---|---|
| FP16 | ~14 GB | Baseline | Reference quality | When you have enough RAM |
| 8-bit | ~7 GB | < 0.5% | Near-lossless | Quality-critical: coding, math |
| 4-bit | ~4 GB | 1–3% | Good — imperceptible in chat | General use — recommended default |
| 3-bit | ~3 GB | 5–8% | Acceptable — occasional errors | Fitting large models in RAM |
| 2-bit | ~2 GB | 15–30% | Noticeable — factual errors | Last resort for memory constraints |

For most users, 4-bit is the correct choice. The 3% perplexity increase from FP16→Q4 is far smaller than the difference between Qwen2.5-7B and Qwen2.5-14B — you'd get more quality improvement by fitting a larger model at 4-bit than running a smaller model at 8-bit.

## When to use pre-quantized mlx-community models

The `mlx-community` organization on Hugging Face maintains pre-converted, pre-quantized versions of most popular models. **Use these instead of converting yourself** unless you have a specific reason not to:

```bash
# These are ready to use immediately — no conversion needed
mlx_lm.generate --model mlx-community/Qwen2.5-7B-Instruct-4bit --prompt "Hello"
mlx_lm.generate --model mlx-community/Qwen2.5-32B-Instruct-4bit --prompt "Hello"
mlx_lm.generate --model mlx-community/Mistral-7B-Instruct-v0.3-4bit --prompt "Hello"
```

**Convert yourself when:**
- The model you want isn't in mlx-community yet (new releases often take a few days)
- You need a non-standard bit depth or group size
- You have a private or fine-tuned model
- You're testing different quantization settings

Browse available models: `https://huggingface.co/mlx-community`

## How to assess quality degradation

Compare the same prompt at different quantization levels to decide if quality is acceptable for your use case:

```python
import subprocess

prompt = "Explain quantum entanglement to a high school student."
models = {
    "4-bit": "./qwen2.5-7b-4bit",
    "8-bit": "./qwen2.5-7b-8bit",
}

for label, model_path in models.items():
    result = subprocess.run(
        ["mlx_lm.generate", "--model", model_path,
         "--prompt", prompt, "--max-tokens", "300"],
        capture_output=True, text=True
    )
    print(f"\n=== {label} ===")
    print(result.stdout)
```

Run this on 3–5 different prompts covering your actual use case before committing to a quantization level.

## Group size tuning

The `--q-group-size` parameter controls how many weights share a scale factor. Smaller groups = more scale factors = more overhead but better accuracy.

```bash
# Default group size (64) — good balance
mlx_lm.convert --hf-path Qwen/Qwen2.5-7B-Instruct \
  --mlx-path ./qwen-q4-g64 --quantize --q-bits 4 --q-group-size 64

# Smaller group size (32) — 10-15% larger file, slightly better quality
mlx_lm.convert --hf-path Qwen/Qwen2.5-7B-Instruct \
  --mlx-path ./qwen-q4-g32 --quantize --q-bits 4 --q-group-size 32
```

Group size 32 is worth trying if you're at 4-bit and noticing quality issues — it often recovers most of the accuracy loss with only a modest size increase.

<!-- tab: Troubleshooting -->
## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Killed` or OOM crash during conversion | Full FP16 model doesn't fit in RAM | Use a machine with more RAM, or use a pre-quantized mlx-community model instead |
| `trust_remote_code` error | Model requires executing custom code from HF | Add `--trust-remote-code` flag to the convert command |
| 403 Forbidden / 401 Unauthorized | Gated model, not logged in or license not accepted | Run `huggingface-cli login`, then accept license on the model's HF page |
| Conversion runs on single CPU thread for hours | Normal behavior — quantization is CPU-bound | Expected: ~5 min for 7B, ~20 min for 32B; progress bar shows per-layer status |
| `mlx_lm.generate` can't find the model | Wrong path or missing config.json | Check the output dir: `ls ./your-model/` should show config.json and model.safetensors |
| Tokenizer errors after quantization | Tokenizer files not copied during conversion | Manually copy from source: `cp ~/.cache/huggingface/hub/.../tokenizer* ./output-dir/` |
| Output file is larger than expected | Group size too small, or model has large embedding layers | Embedding layers aren't always quantized; use `--q-bits 8` for embeddings if needed |
| `No model.safetensors in repository` | Model uses sharded files (model-00001-of-00004.safetensors etc.) | This is handled automatically by mlx_lm.convert — it downloads and merges shards |

### Diagnosing RAM requirements before starting

Before running a conversion, estimate whether it will fit in your machine's RAM:

```python
# Estimate RAM needed for FP16 conversion
def estimate_conversion_ram_gb(param_billions: float) -> dict:
    fp16_gb = param_billions * 2      # 2 bytes per parameter
    return {
        "source_model_fp16_gb": fp16_gb,
        "output_4bit_gb": param_billions * 0.5,
        "peak_ram_gb": fp16_gb * 1.2,  # 20% overhead for intermediate tensors
        "fits_on_32gb": fp16_gb * 1.2 < 32,
    }

print(estimate_conversion_ram_gb(7))    # 7B model
# {'source_model_fp16_gb': 14.0, 'output_4bit_gb': 3.5, 'peak_ram_gb': 16.8, 'fits_on_32gb': True}
print(estimate_conversion_ram_gb(32))   # 32B model
# {'source_model_fp16_gb': 64.0, 'output_4bit_gb': 16.0, 'peak_ram_gb': 76.8, 'fits_on_32gb': False}
```

For models that don't fit in RAM for conversion, use the pre-quantized version from mlx-community.

### Fixing trust_remote_code errors

Some models (Phi-3, some RWKV variants) require executing Python code from the model's repository to load the architecture. MLX requires explicit opt-in for security reasons:

```bash
# Add --trust-remote-code to allow executing model-specific code
mlx_lm.convert \
  --hf-path microsoft/Phi-3-mini-4k-instruct \
  --mlx-path ./phi3-mini-4bit \
  --quantize --q-bits 4 \
  --trust-remote-code               # only add this for models that require it
```

Only use `--trust-remote-code` for models from organizations you trust. The code runs on your machine with your permissions.

### Handling gated model 403 errors

```bash
# Step 1: Log in
huggingface-cli login

# Step 2: Accept the license on the web for the specific model
# Example for Llama-3: https://huggingface.co/meta-llama/Meta-Llama-3-8B
# Click "Agree and access repository" — approval is usually instant

# Step 3: Verify access
huggingface-cli download meta-llama/Meta-Llama-3-8B config.json   # should succeed

# Step 4: Run conversion
mlx_lm.convert --hf-path meta-llama/Meta-Llama-3-8B-Instruct \
  --mlx-path ./llama3-8b-4bit --quantize --q-bits 4
```
