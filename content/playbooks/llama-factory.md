---
slug: llama-factory
title: "LLaMA Factory on macOS"
time: "1 hr"
color: red
desc: "Install and fine-tune models with LLaMA Factory on MPS"
tags: [fine-tuning, ui]
spark: "LLaMA Factory"
category: fine-tuning
featured: false
whatsNew: false
---

<!-- tab: Overview -->
## Basic idea

LLaMA Factory is a unified fine-tuning framework that supports 100+ model architectures and multiple training paradigms — SFT (Supervised Fine-Tuning), DPO (Direct Preference Optimization), ORPO, and PPO — through a single WebUI or CLI. Rather than writing training code from scratch, you configure a run through a form, click Start, and LLaMA Factory handles the rest.

On macOS, it routes all computation through the PyTorch MPS backend, so all the MPS constraints apply (float32 only, fallback env var for missing ops). LLaMA Factory's main advantage over writing raw HuggingFace Trainer code is its built-in support for DPO and RLHF workflows, which require substantial infrastructure to implement manually.

## What you'll accomplish

LLaMA Factory installed and serving its WebUI at `localhost:7860`, a completed LoRA fine-tune of Qwen2.5-7B-Instruct on the built-in `alpaca_en_demo` dataset (~50K samples), and the adapter saved to disk ready for inference or further evaluation.

## What to know before starting

- **SFT vs DPO**: SFT trains the model to predict the next token given an input — it learns "what to say." DPO trains from preference pairs (chosen vs rejected responses), teaching the model "what's better." LLaMA Factory supports both with the same WebUI.
- **LLaMA Factory wraps HuggingFace Trainer**: All MPS limitations apply. Float16 training will produce NaN loss. The `PYTORCH_ENABLE_MPS_FALLBACK=1` environment variable must be set.
- **Dataset formats**: LLaMA Factory supports two main formats. Alpaca format has `instruction`, `input`, `output` fields. ShareGPT format has a `conversations` list with `from`/`value` pairs. Using the wrong format produces empty training batches.
- **WebUI maps to CLI arguments 1:1**: Every WebUI field corresponds to a YAML config key. Once you've found working settings in the WebUI, export them to YAML for reproducible CLI runs.

## Prerequisites

- macOS 12.3 or later
- Apple Silicon Mac (M1, M2, or M3 family)
- Python 3.9 or later
- git installed
- 16 GB+ unified memory (32 GB recommended for 7B models)
- ~20 GB free disk space for model weights and checkpoints

## Time & risk

- **Duration**: ~1 hour (install 15 min, model download 20-30 min, training varies)
- **Risk level**: Medium — clones a large repository and installs dozens of packages. Use a venv.
- **Rollback**: Deactivate and delete the venv, delete the cloned repository and downloaded model cache.

<!-- tab: Install -->
## Step 1: Clone the repository

Installing from source rather than PyPI gives you access to the latest macOS-specific fixes and features. The main LLaMA Factory PyPI package often lags the source by weeks.

```bash
# Clone to your projects directory — the folder will be named LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git ~/LLaMA-Factory
cd ~/LLaMA-Factory

# Check the latest release tag (optional — main branch is generally stable)
git log --oneline -5
```

## Step 2: Create a venv and install dependencies

The `[torch,metrics]` extras install PyTorch with MPS support and evaluation libraries (ROUGE, BLEU). The `-e` (editable) flag means Python imports the package directly from the cloned directory — useful for hacking on the source or pulling updates with `git pull`.

```bash
# Create isolated environment
python3 -m venv ~/.venvs/llama-factory
source ~/.venvs/llama-factory/bin/activate

# Install LLaMA Factory with PyTorch and evaluation metric dependencies
# [torch]    = PyTorch + torchvision (includes MPS support)
# [metrics]  = rouge-score, nltk, jieba for BLEU/ROUGE evaluation
pip install -e ".[torch,metrics]"

# Verify the CLI is installed
llamafactory-cli version
```

## Step 3: Launch the WebUI

The WebUI serves on all network interfaces by default (`0.0.0.0:7860`), making it accessible from other devices on your local network. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` so operations not yet implemented in Metal fall back to CPU automatically.

```bash
# Enable MPS fallback globally for this session
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Launch the WebUI
llamafactory-cli webui
```

Open `http://localhost:7860` in your browser. If you see a Gradio interface with a "Model Name" field, the WebUI is running correctly. To access from another machine on the same network, use your Mac's local IP address: `http://192.168.x.x:7860`.

## Step 4: Verify MPS is selected in the WebUI

In the WebUI, scroll to the **Training Device** field. It should default to `mps` on Apple Silicon. If it shows `cuda` or `cpu`, change it manually. A quick way to confirm before committing to a full training run:

```bash
# CLI verification — print device info without starting training
python -c "
import torch
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'Default device will be: mps' if torch.backends.mps.is_available() else 'cpu')
"
```

<!-- tab: Fine-tune -->
## Step 1: Enter the model name in the WebUI

In the **Model Name** field, enter `Qwen/Qwen2.5-7B-Instruct`. LLaMA Factory downloads this from HuggingFace automatically on the first run and caches it at `~/.cache/huggingface/hub`. Subsequent runs use the cached version.

Alternatively, specify a local directory path if you've already downloaded the model: `/path/to/Qwen2.5-7B-Instruct`.

The model variant matters: `-Instruct` models have instruction-following fine-tuning already applied. For SFT on a custom dataset, start with an instruct model — it converges faster than training a base model.

## Step 2: Select training method and dataset

- **Finetuning Method**: Select `LoRA`. Full fine-tuning requires 3-4x more memory and produces larger output files. Freeze fine-tuning trains only the top N transformer layers — a middle ground, rarely the best choice.
- **Dataset**: Select `alpaca_en_demo` for a quick test. This is a 50K sample subset of the Alpaca dataset in alpaca format (instruction/input/output fields), included with LLaMA Factory.
- **Template**: Select `qwen` to match the model's chat format.

## Step 3: Configure hyperparameters for MPS

These settings are necessary for stable training on the MPS backend:

| Field | Value | Why |
|---|---|---|
| Training Device | `mps` | Route compute to Apple Silicon GPU |
| FP16 | OFF | MPS fp16 backward pass produces NaN |
| Per Device Batch Size | `1` | MPS memory limit with 7B model |
| Gradient Accumulation | `8` | Effective batch = 8, mimics larger GPU |
| LoRA Rank | `8` | Start conservative, increase if underfitting |
| Learning Rate | `1e-4` | Standard starting point for LoRA |
| Epochs | `3` | Enough for convergence on alpaca_en_demo |

## Step 4: Start training and monitor the log tab

Click **Start** and switch to the **Log** tab. You should see loss values printed every N steps. A healthy training run looks like:
- Loss starts ~2.0-3.0 on step 1
- Loss decreases to ~1.0-1.5 over the first epoch
- Loss continues decreasing or plateaus in later epochs

To stop safely, click **Abort** in the WebUI — this saves the current checkpoint before exiting. Killing the terminal abruptly may corrupt the in-progress checkpoint.

## Step 5: CLI alternative with YAML config

The WebUI is great for exploration, but for reproducible runs, export your settings to YAML and use the CLI. This allows version-controlling your training configuration:

```yaml
# Save as qwen-lora-mps.yaml
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
lora_target: q_proj,v_proj

dataset: alpaca_en_demo
template: qwen
cutoff_len: 1024
max_samples: 1000            # Limit to 1000 samples for a quick test run

output_dir: ./saves/qwen-lora-mps
logging_steps: 10
save_steps: 100
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
fp16: false                  # Required for MPS
```

```bash
# Run from the LLaMA-Factory directory with MPS fallback enabled
PYTORCH_ENABLE_MPS_FALLBACK=1 llamafactory-cli train qwen-lora-mps.yaml
```

<!-- tab: Troubleshooting -->
## Common issues

| Symptom | Cause | Fix |
|---|---|---|
| `Error: CUDA is not available` | Device set to `cuda` | Change Training Device to `mps` in WebUI, or set `device: mps` in YAML |
| WebUI crashes immediately after launch | Port 7860 already in use | Kill the other process (`lsof -ti:7860 \| xargs kill`) or pass `--port 7861` |
| Model download hangs or fails | HuggingFace rate limiting or auth required | Run `huggingface-cli login` and retry; some models require accepting a license |
| `RuntimeError: MPS backend out of memory` | 7B model too large for available memory | Use a 3B model (e.g., Qwen2.5-3B-Instruct), or reduce `cutoff_len` to 512 |
| `NotImplementedError: operator not implemented for MPS` | Metal missing a required op | Set `PYTORCH_ENABLE_MPS_FALLBACK=1` before launching; restart WebUI |
| `KeyError: 'instruction'` or empty batches | Dataset format mismatch | Verify your dataset uses alpaca format; switch to `alpaca_en_demo` to test |
| Training stops and no checkpoint is saved | Killed before first `save_steps` checkpoint | Lower `save_steps` to 50; use **Abort** button instead of killing the process |

## Port conflict: finding and killing the blocking process

```bash
# Find what is using port 7860
lsof -i :7860

# Kill it by PID (replace 12345 with the actual PID from lsof output)
kill 12345

# Or kill everything on that port at once
lsof -ti:7860 | xargs kill -9

# Relaunch on a different port if you want to keep both processes
llamafactory-cli webui --port 7861
```

## HuggingFace authentication for gated models

Some models (Llama 3, Gemma) require accepting a license and authenticating:

```bash
# Install the HuggingFace CLI if not present
pip install huggingface_hub

# Log in with your HuggingFace token (get one at huggingface.co/settings/tokens)
huggingface-cli login

# Verify authentication worked
huggingface-cli whoami

# Then re-run training — the download will proceed automatically
PYTORCH_ENABLE_MPS_FALLBACK=1 llamafactory-cli train your-config.yaml
```

## MPS fallback is silently making training very slow

If training is much slower than expected (>10 min/step for a 7B model), many ops are likely falling back to CPU:

```bash
# Run with verbose fallback logging and grep for fallback messages
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \
  llamafactory-cli train your-config.yaml 2>&1 | tee train.log | grep -i fallback

# Count how many fallbacks are occurring
grep -c "fallback" train.log
```

If the fallback count is high (>100 per step), the MPS backend is not a good fit for this model/config combination. Consider switching to full CPU training (`device: cpu`) or using MLX-based fine-tuning instead.
