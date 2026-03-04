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

LLaMA Factory is a unified fine-tuning framework with a web UI that supports many model architectures. It works on macOS with the PyTorch MPS backend.

## Prerequisites

- macOS 12.3+
- Apple Silicon Mac
- Python 3.9+
- 32 GB+ unified memory recommended

## Time & risk

- **Duration:** 1 hour
- **Risk level:** Medium
- **Note:** Not all features work on MPS — some operations fall back to CPU

<!-- tab: Install -->
## Clone and install

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

## Launch the web UI

```bash
llamafactory-cli webui
```

Open http://localhost:7860 in your browser.

<!-- tab: Fine-tune -->
## Web UI workflow

1. **Model Name**: Enter a Hugging Face model (e.g., `Qwen/Qwen2.5-7B-Instruct`)
2. **Finetuning Method**: Select `LoRA`
3. **Dataset**: Select or upload your dataset
4. **Training Device**: Select `mps`
5. Click **Start** to begin training

## CLI fine-tuning

```bash
llamafactory-cli train \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --stage sft \
  --do_train \
  --finetuning_type lora \
  --lora_rank 8 \
  --dataset alpaca_en_demo \
  --template qwen \
  --output_dir ./saves/qwen-lora \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --lr_scheduler_type cosine \
  --learning_rate 1e-4 \
  --num_train_epochs 3.0
```

<!-- tab: Troubleshooting -->
## CUDA not available error

Ensure you set the device to `mps` in the configuration. Some modules check for CUDA explicitly.

## Web UI crashes

Try the CLI instead of the web UI for more stability on macOS.

## Memory errors

Use a smaller base model (7B instead of 13B+), reduce batch size, and use gradient checkpointing.
