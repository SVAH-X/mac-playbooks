---
slug: pytorch-mps
title: "PyTorch MPS Fine-tuning"
time: "1 hr"
color: red
desc: "GPU-accelerated training using PyTorch Metal backend"
tags: [pytorch, fine-tuning]
spark: "Fine-tune with PyTorch"
category: fine-tuning
featured: false
whatsNew: false
---

<!-- tab: Overview -->
## Basic idea

PyTorch's MPS (Metal Performance Shaders) backend provides GPU acceleration on Apple Silicon for PyTorch workloads, including model fine-tuning.

## Note on performance

PyTorch MPS is functional but significantly slower than MLX for most Apple Silicon fine-tuning workloads. Prefer MLX LoRA fine-tuning unless you need PyTorch-specific features (custom training loops, specific optimizers, PEFT integration, etc.).

## Prerequisites

- macOS 12.3+
- Apple Silicon Mac
- Python 3.9+
- Xcode Command Line Tools

## Time & risk

- **Duration:** 1 hour (including training)
- **Risk level:** Medium — requires more memory than MLX

<!-- tab: Setup -->
## Install dependencies

```bash
pip install torch torchvision torchaudio
pip install transformers datasets accelerate peft
```

## Verify MPS is available

```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
```

<!-- tab: Fine-tune -->
## Full fine-tuning with Trainer

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

device = torch.device("mps")

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.float32  # MPS requires float32 for training
).to(device)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    fp16=False,         # MPS doesn't support fp16 training
    use_mps_device=True,
    save_strategy="epoch",
    logging_steps=10,
)
```

## LoRA with PEFT on MPS

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

<!-- tab: Troubleshooting -->
## RuntimeError: MPS backend out of memory

Reduce batch size to 1, increase gradient_accumulation_steps.

## NotImplementedError: certain ops

Some PyTorch operations aren't implemented on MPS yet. Add `PYTORCH_ENABLE_MPS_FALLBACK=1` to fall back to CPU for those ops:
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python train.py
```

## fp16 training errors

MPS does not support fp16 training. Use float32 (`fp16=False`).
