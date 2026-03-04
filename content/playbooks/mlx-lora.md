---
slug: mlx-lora
title: "MLX LoRA Fine-tuning"
time: "30 min"
color: orange
desc: "Fine-tune LLMs with LoRA/QLoRA natively on Apple Silicon"
tags: [mlx, fine-tuning]
spark: "NeMo / Unsloth Fine-tune"
category: fine-tuning
featured: false
whatsNew: false
---

<!-- tab: Overview -->
## Basic idea

MLX LM supports LoRA (Low-Rank Adaptation) fine-tuning natively on Apple Silicon. Fine-tune a 7B model with custom data in 30 minutes on a 32GB Mac.

## Why MLX over Unsloth on Mac?

- Unsloth is CUDA-only and doesn't run on macOS
- MLX LoRA achieves comparable throughput to Unsloth on NVIDIA for Apple Silicon
- No special setup required — just `pip install mlx-lm`

## Prerequisites

- macOS 14.0+
- Apple Silicon Mac
- Python 3.10+
- 16 GB+ unified memory (32 GB recommended for 7B models)

## Time & risk

- **Duration:** 30 minutes (plus training time — ~20 min for 1000 iters on M3 Max)
- **Risk level:** Low — creates adapter files, original model untouched

<!-- tab: Prepare Data -->
## Data format

Training data should be JSONL with a `text` field containing the formatted conversation:

```json
{"text": "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n4<|im_end|>"}
{"text": "<|im_start|>user\nCapital of France?<|im_end|>\n<|im_start|>assistant\nParis<|im_end|>"}
```

## Directory structure

```
training_data/
├── train.jsonl
└── valid.jsonl
```

## Prepare a simple dataset

```python
import json

train_data = [
    {"text": "<|im_start|>user\nWhat is MLX?<|im_end|>\n<|im_start|>assistant\nMLX is Apple's machine learning framework for Apple Silicon.<|im_end|>"},
    # Add more examples...
]

with open("training_data/train.jsonl", "w") as f:
    for item in train_data:
        f.write(json.dumps(item) + "\n")
```

<!-- tab: Fine-tune -->
## Run LoRA fine-tuning

```bash
mlx_lm.lora \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --data ./training_data \
  --train \
  --iters 1000 \
  --learning-rate 1e-5 \
  --batch-size 4 \
  --lora-layers 16 \
  --adapter-path ./adapters
```

## Monitor training

The command prints loss and tokens/sec every N iterations. Watch for loss to decrease steadily.

## Test the adapter

```bash
mlx_lm.generate \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --adapter-path ./adapters \
  --prompt "What is MLX?"
```

<!-- tab: Export -->
## Merge adapter into model

```bash
mlx_lm.fuse \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --adapter-path ./adapters \
  --save-path ./fused-model
```

## Run the merged model

```bash
mlx_lm.generate --model ./fused-model --prompt "Test prompt"
```

## Convert to GGUF for Ollama

```bash
# Install llama.cpp conversion tools
pip install gguf
python llama.cpp/convert_hf_to_gguf.py ./fused-model --outfile ./fused-model.gguf

# Use with Ollama
ollama create my-model -f ./Modelfile
```
