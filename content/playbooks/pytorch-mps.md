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

PyTorch's MPS (Metal Performance Shaders) backend lets PyTorch dispatch tensor operations to Apple Silicon's GPU instead of the CPU. When you call `.to(device("mps"))` on a tensor, PyTorch routes all computations for that tensor through Metal compute shaders — the same GPU API that powers macOS graphics. This is the standard path for HuggingFace Trainer-based fine-tuning workflows that don't have direct MLX equivalents, such as PEFT/LoRA training with custom datasets and callbacks.

MPS is functional but slower than MLX for most workflows. Use it when you need PyTorch-specific libraries (PEFT, DeepSpeed, custom training loops) that haven't been ported to MLX.

## What you'll accomplish

A working PyTorch MPS fine-tuning environment on your Mac's GPU. You will LoRA fine-tune a 7B parameter causal language model using HuggingFace PEFT and Trainer, with checkpoints saved to disk and a verified training run showing decreasing loss.

## What to know before starting

- **Device placement**: PyTorch tensors must be on the same device to operate on each other. If the model is on MPS and the labels are on CPU, you get a runtime error — every tensor in a batch must be explicitly moved.
- **Gradient accumulation**: Simulates a larger batch size by running `N` forward passes and summing gradients before each optimizer step. `per_device_train_batch_size=1` with `gradient_accumulation_steps=8` gives an effective batch size of 8 without needing 8x the memory.
- **PEFT/LoRA**: Instead of updating all model weights, LoRA injects small trainable rank-decomposition matrices into specific layers. Only ~0.1% of parameters are trained, keeping memory and compute manageable.
- **MPS operation gaps**: Metal lacks some CUDA primitives. When PyTorch encounters an unimplemented op on MPS, it either raises `NotImplementedError` or silently falls back to CPU depending on your environment variable setting.
- **float32 requirement**: MPS cannot reliably execute the fp16 backward pass — gradients overflow or go NaN. You must use `torch_dtype=torch.float32` and `fp16=False` in TrainingArguments. This roughly doubles memory usage compared to CUDA fp16 training.

## Prerequisites

- macOS 12.3 or later (MPS was introduced in 12.3)
- Apple Silicon Mac (M1, M2, or M3 family)
- Python 3.9 or later
- Xcode Command Line Tools installed (`xcode-select --install`)
- 16 GB+ unified memory (32 GB recommended for 7B models)

## Time & risk

- **Duration**: ~1 hour (setup 20 min, training varies by dataset size)
- **Risk level**: Medium — ML dependencies have strict version requirements and pip can produce conflicting installs. Always use a virtual environment.
- **Rollback**: Delete the venv directory to undo all changes. Model checkpoints are self-contained.

<!-- tab: Setup -->
## Step 1: Create a virtual environment

Isolation is critical for ML stacks. `torch`, `transformers`, `peft`, and `accelerate` all pin their dependencies tightly, and installing them into your system Python almost always produces version conflicts. A venv keeps this install self-contained and deletable.

```bash
# Create a dedicated environment for this project
python3 -m venv ~/.venvs/pytorch-mps

# Activate it — you must do this in every new terminal session
source ~/.venvs/pytorch-mps/bin/activate

# Confirm the active interpreter
which python  # Should show ~/.venvs/pytorch-mps/bin/python
```

## Step 2: Install PyTorch with MPS support

On macOS, the standard `torch` wheel includes MPS support — there are no separate CUDA wheels to avoid. PyTorch >= 2.0 is required for stable MPS training.

```bash
# Install PyTorch — the macOS wheel includes MPS backend automatically
pip install torch torchvision torchaudio

# Verify MPS is available before proceeding
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available:   {torch.backends.mps.is_available()}')
print(f'MPS built:       {torch.backends.mps.is_built()}')

# Quick smoke test — create a tensor directly on MPS
t = torch.tensor([1.0, 2.0, 3.0], device='mps')
print(f'Tensor device:   {t.device}')  # Should print: mps:0
"
```

Both `is_available()` and `is_built()` must return `True`. If `is_built()` is `True` but `is_available()` is `False`, your macOS version is below 12.3.

## Step 3: Install the HuggingFace stack

Each package has a specific role in the training pipeline:

```bash
# transformers: model architectures, tokenizers, and the Trainer class
# peft: LoRA and other parameter-efficient fine-tuning methods
# datasets: efficient data loading and tokenization pipelines
# accelerate: distributed training abstractions (Trainer uses this internally)
pip install transformers peft datasets accelerate
```

Verify the installs:

```bash
python -c "
import transformers, peft, datasets, accelerate
print(f'transformers: {transformers.__version__}')
print(f'peft:         {peft.__version__}')
print(f'datasets:     {datasets.__version__}')
print(f'accelerate:   {accelerate.__version__}')
"
```

## Step 4: Verify MPS with a real tensor operation

A smoke test that confirms the MPS backend handles an actual computation end-to-end:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python -c "
import torch

device = torch.device('mps')

# Create two matrices on MPS and multiply them
a = torch.randn(1000, 1000, device=device, dtype=torch.float32)
b = torch.randn(1000, 1000, device=device, dtype=torch.float32)
c = a @ b

# Confirm the result lives on MPS
print(f'Result device: {c.device}')   # mps:0
print(f'Result shape:  {c.shape}')    # torch.Size([1000, 1000])
print('MPS matmul: OK')
"
```

Setting `PYTORCH_ENABLE_MPS_FALLBACK=1` enables automatic CPU fallback for unimplemented ops — keep this set for all training runs.

<!-- tab: Fine-tune -->
## Step 1: Load the model with MPS placement

The `torch_dtype=torch.float32` argument is not optional on MPS — float16 causes NaN gradients during the backward pass. This doubles memory consumption compared to fp16 CUDA training, which is why 16 GB minimum RAM is required for a 7B model.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python << 'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
device = torch.device("mps")

# float32 is required — float16 backward pass is broken on MPS
# This will download ~14GB on first run; subsequent runs use the cache
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,  # NOT float16
    low_cpu_mem_usage=True,     # Load layer-by-layer to avoid peak RAM spike
)
model = model.to(device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token  # Required for batch padding

print(f"Model device: {next(model.parameters()).device}")  # mps:0
print(f"Model dtype:  {next(model.parameters()).dtype}")   # torch.float32
EOF
```

## Step 2: Configure LoRA adapters

LoRA injects trainable matrices into the attention projection layers. The key parameters:
- `r=16`: rank of the adapter matrices — higher rank = more capacity = more memory
- `lora_alpha=32`: scaling factor; effective learning rate = `lora_alpha / r`
- `target_modules`: which linear layers to inject into — q_proj/v_proj is a common starting point

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                          # Adapter rank — start here, tune up if quality is poor
    lora_alpha=32,                 # Scaling: lora_alpha/r = 2.0 effective LR multiplier
    target_modules=["q_proj", "v_proj"],  # Attention query and value projections
    lora_dropout=0.05,             # Dropout on adapter layers to prevent overfitting
    bias="none",                   # Don't train bias terms (saves memory)
    task_type="CAUSAL_LM"          # Tells PEFT we're doing causal language modeling
)

model = get_peft_model(model, lora_config)

# This line shows exactly how many parameters are being trained
# Expect something like: trainable params: 41,943,040 || all params: 7,622,230,016 || trainable%: 0.55
model.print_trainable_parameters()
```

## Step 3: Set up TrainingArguments for MPS

Each argument has a specific reason related to MPS constraints:

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./output",                   # Where checkpoints are saved
    num_train_epochs=3,
    per_device_train_batch_size=1,           # MPS has limited memory — keep at 1
    gradient_accumulation_steps=8,           # Effective batch size = 1 * 8 = 8
    learning_rate=2e-4,                      # Higher LR is OK for LoRA (fewer params)
    fp16=False,                              # MUST be False — MPS can't do fp16 backward
    bf16=False,                              # BF16 also unsupported on MPS
    use_mps_device=True,                     # Tell Trainer to use MPS explicitly
    save_strategy="steps",                   # Save a checkpoint every N steps
    save_steps=100,
    logging_steps=10,                        # Print loss every 10 steps
    gradient_checkpointing=True,             # Trade compute for memory (recomputes activations)
    dataloader_num_workers=0,                # MPS + multiprocessing can deadlock — use 0
    report_to="none",                        # Disable W&B/TensorBoard for this example
)
```

## Step 4: Train and monitor

Start training. The loss should decrease over time. If it plateaus early or spikes, see Troubleshooting.

```python
from transformers import Trainer
from datasets import load_dataset

# Load a small demonstration dataset
dataset = load_dataset("tatsu-lab/alpaca", split="train[:500]")

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )

tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
)

# Begin training — watch the 'loss' column in the log output
# loss should decrease from ~2-3 down toward ~1.0 over the run
trainer.train()

# To resume from a checkpoint after interruption:
# trainer.train(resume_from_checkpoint="./output/checkpoint-100")
```

Watch for `loss` decreasing and `grad_norm` remaining stable (not exploding above ~10). If loss goes NaN, check that `fp16=False` is set.

<!-- tab: Troubleshooting -->
## Common issues

| Symptom | Cause | Fix |
|---|---|---|
| `NotImplementedError: The operator ... is not currently implemented for the MPS device` | Metal lacks this op | Set `PYTORCH_ENABLE_MPS_FALLBACK=1` as an env var before running |
| Loss is `NaN` from step 1 | fp16 is enabled on MPS | Set `fp16=False` and `bf16=False` in TrainingArguments |
| `RuntimeError: MPS backend out of memory` | GPU memory exhausted | Reduce `per_device_train_batch_size`, increase `gradient_accumulation_steps`, enable `gradient_checkpointing=True` |
| Training is slower than expected | Ops fell back to CPU silently | Run with `PYTORCH_ENABLE_MPS_FALLBACK=1` and watch for "fallback" warnings in output |
| `MPS backend not available` | macOS version < 12.3 | Upgrade macOS or run on CPU (`device = "cpu"`) |
| Checkpoint file is 0 bytes or corrupt | Training interrupted mid-save | Delete the corrupt checkpoint directory, resume from the previous one |
| `ValueError: Asking to pad but the tokenizer does not have a padding token` | Tokenizer missing pad token | Add `tokenizer.pad_token = tokenizer.eos_token` before tokenizing |

## MPS fallback: finding which ops fall back to CPU

When ops fall back to CPU, training silently slows down. To identify which ops are falling back:

```bash
# Run with fallback logging enabled
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python train.py 2>&1 | grep -i "fallback\|mps"
```

If you see many fallback messages, the model may train faster on CPU altogether. Benchmark both:

```python
import torch, time

device_mps = torch.device("mps")
device_cpu = torch.device("cpu")

x = torch.randn(1000, 1000, dtype=torch.float32)

for dev in [device_cpu, device_mps]:
    t = x.to(dev)
    start = time.perf_counter()
    for _ in range(100):
        _ = t @ t
    if dev.type == "mps":
        torch.mps.synchronize()  # MPS is async — must sync before timing
    elapsed = time.perf_counter() - start
    print(f"{dev}: {elapsed:.3f}s")
```

## OOM: reducing memory usage

If you hit out-of-memory errors, apply these in order:

```python
# Option 1: Enable gradient checkpointing (trades 30% more compute for ~40% less memory)
model.gradient_checkpointing_enable()

# Option 2: Reduce LoRA rank (less memory, less model capacity)
lora_config = LoraConfig(r=8, ...)  # Down from r=16

# Option 3: Reduce sequence length
tokenizer(..., max_length=256)  # Down from 512

# Option 4: Free MPS cache between runs
import torch
torch.mps.empty_cache()
```

## Checkpoint corruption on interruption

If you kill training mid-step, the checkpoint being written at that moment will be corrupt. The previous checkpoint is intact:

```bash
# List checkpoints sorted by modification time
ls -lt ./output/checkpoint-*/trainer_state.json

# Verify a checkpoint is readable
python -c "
from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch
m = AutoModelForCausalLM.from_pretrained('./output/checkpoint-100', torch_dtype=torch.float32)
print('Checkpoint OK')
"
```
