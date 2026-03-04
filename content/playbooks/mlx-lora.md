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

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that adds small trainable matrices to a frozen pretrained model. Instead of updating all 7 billion parameters — which would require more GPU memory than a Mac has — LoRA inserts pairs of small matrices (A and B) at each attention layer. Only these adapter matrices are updated during training. The frozen model weights never change.

The math: a standard weight matrix W has shape [4096, 4096] = 16.7M parameters. LoRA replaces the weight update ΔW with two smaller matrices: A [4096, r] and B [r, 4096], where r is the "rank" (typically 4–16). At rank 8, this is [4096×8] + [8×4096] = 65,536 parameters — just 0.4% of the original. During inference, the adapter is applied as: output = W·x + (B·A)·x·scaling_factor.

On Apple Silicon, MLX implements LoRA natively using Metal: the frozen model runs on the GPU, and gradients flow through only the adapter matrices, keeping memory usage manageable even on 16 GB machines.

## What you'll accomplish

A LoRA adapter fine-tuned on your custom dataset (100–10,000 examples), tested for quality improvement over the base model, and merged into a standalone model ready for deployment with `mlx_lm.generate` or converted to GGUF for Ollama.

## What to know before starting

- **Fine-tuning vs prompting** — prompting changes how you ask; fine-tuning changes what the model knows. Use fine-tuning when you need consistent style, specialized vocabulary, a specific response format, or behavior that system prompts can't reliably produce.
- **LoRA rank** — `r=4` adds ~0.1% of parameters, good for style adaptation. `r=16` adds ~0.5%, better for learning new facts or formats. Higher rank = more capacity but also more memory and training time. Start with `r=8` for general use.
- **Chat template** — instruct models expect inputs wrapped in special tokens that signal role boundaries. Qwen2.5 uses ChatML format: `<|im_start|>user\nYour message<|im_end|>\n<|im_start|>assistant\n`. If your training data doesn't match the model's expected template, training will proceed but results will be poor.
- **Learning rate** — how much the weights change per gradient step. Too high (> 5e-4) causes loss spikes; too low (< 1e-6) means no learning. The safe range for LoRA is 1e-5 to 2e-4. Default `1e-5` is conservative and safe.
- **Iterations vs epochs** — MLX counts training steps (gradient updates), not passes through the data. With batch_size=4 and 400 training examples, one epoch = 100 iterations. 1000 iterations = 10 epochs.

## Prerequisites

- macOS 14.0+ (Sonoma or later)
- Apple Silicon Mac (M1 or later)
- Python 3.10, 3.11, or 3.12
- `mlx-lm` installed: `pip install mlx-lm`
- 16 GB+ unified memory (32 GB recommended for 7B full-quality fine-tuning)
- Custom dataset in JSONL format (see Prepare Data tab for the required structure)

## Time & risk

- **Duration:** 30 minutes setup + training time (1000 iters ≈ 20 min for 7B on M3 Max, ≈ 45 min on M1 Pro)
- **Risk level:** Low — adapter files are stored separately; the base model is never modified
- **Rollback:** Delete `./adapters/` directory; the base model in `~/.cache/huggingface/` is unchanged

<!-- tab: Prepare Data -->
## Step 1: Understand the data format

MLX LoRA expects a JSONL file where each line is a JSON object with a `"text"` field containing a complete, formatted conversation. The format must exactly match the model's training format — Qwen2.5-Instruct uses the ChatML template.

```json
{"text": "<|im_start|>user\nWhat is the boiling point of water?<|im_end|>\n<|im_start|>assistant\nThe boiling point of water at sea level is 100°C (212°F).<|im_end|>"}
{"text": "<|im_start|>user\nConvert 72°F to Celsius.<|im_end|>\n<|im_start|>assistant\nTo convert 72°F to Celsius: (72 - 32) × 5/9 = 22.2°C<|im_end|>"}
{"text": "<|im_start|>system\nYou are a helpful cooking assistant.<|im_end|>\n<|im_start|>user\nHow long should I boil pasta?<|im_end|>\n<|im_start|>assistant\nBoil pasta for 8–12 minutes depending on the type, until al dente. Check the package instructions and taste-test near the end.<|im_end|>"}
```

Key formatting rules:
- Each turn ends with `<|im_end|>\n` — the newline is required
- The `system` turn (if any) comes before the first `user` turn
- The final `assistant` turn must NOT end with `<|im_end|>` in some templates — check your model's tokenizer_config.json
- The entire conversation (all turns) goes in a single `"text"` field on one line

To confirm which template your model uses:

```bash
# Check the chat template stored in the tokenizer config
python3 -c "
import json
from pathlib import Path
from huggingface_hub import hf_hub_download

# Download just the tokenizer config
path = hf_hub_download('mlx-community/Qwen2.5-7B-Instruct-4bit', 'tokenizer_config.json')
config = json.loads(Path(path).read_text())
print(config.get('chat_template', 'No template found'))
"
```

## Step 2: Create train/valid split

Always set aside a validation set before training. The validation loss tells you whether the model is learning generalizable patterns or just memorizing training examples — the crucial distinction between good fine-tuning and overfitting.

```
data/
├── train.jsonl     # 90% of your examples — what the model learns from
└── valid.jsonl     # 10% of your examples — never seen during training, used to measure generalization
```

Rules of thumb:
- Minimum viable dataset: 50 examples (valid quality, not robust)
- Recommended minimum: 500 examples
- Validation set size: 10% of total, minimum 25 examples
- For 100-example datasets: 90 train / 10 valid is acceptable

```bash
# Check your splits
wc -l data/train.jsonl data/valid.jsonl
# Expected output:
# 450 data/train.jsonl
#  50 data/valid.jsonl
# 500 total
```

## Step 3: Format your data programmatically

If your data is in another format (CSV, plain text, structured JSON), use this script to convert it to the required JSONL with the correct chat template.

```python
import json
import random
from pathlib import Path

# Your raw data — replace with actual loading logic
raw_examples = [
    {"question": "What is 2+2?", "answer": "4"},
    {"question": "What is the capital of Japan?", "answer": "Tokyo"},
    # ... your data here
]

def format_as_chatml(question: str, answer: str, system: str = None) -> str:
    """Format a Q&A pair as a ChatML conversation for Qwen2.5-Instruct."""
    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>")
    parts.append(f"<|im_start|>user\n{question}<|im_end|>")
    parts.append(f"<|im_start|>assistant\n{answer}<|im_end|>")
    return "\n".join(parts)

# Shuffle and split
random.shuffle(raw_examples)
split_idx = int(len(raw_examples) * 0.9)
train_examples = raw_examples[:split_idx]
valid_examples = raw_examples[split_idx:]

# Write JSONL files
Path("data").mkdir(exist_ok=True)

with open("data/train.jsonl", "w") as f:
    for ex in train_examples:
        text = format_as_chatml(ex["question"], ex["answer"])
        f.write(json.dumps({"text": text}) + "\n")

with open("data/valid.jsonl", "w") as f:
    for ex in valid_examples:
        text = format_as_chatml(ex["question"], ex["answer"])
        f.write(json.dumps({"text": text}) + "\n")

print(f"Train: {len(train_examples)} examples")
print(f"Valid: {len(valid_examples)} examples")
```

## Step 4: Validate the dataset format

Before starting a long training run, validate that the data loads correctly. This catches format errors that would otherwise cause cryptic failures after 10 minutes of waiting.

```bash
# Dry-run validation: load data, tokenize, check for issues, then stop
# --val-batches 1 runs just one validation step to confirm the data pipeline works
mlx_lm.lora \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --data ./data \
  --iters 1 \                                         # just 1 training step
  --val-batches 1 \                                   # just 1 validation step
  --adapter-path ./adapters-test                      # temporary path for validation run
```

A successful validation run prints loss values without errors. Common errors at this stage: `KeyError: 'text'` (wrong field name in your JSONL), `ValueError: sequence length` (a single example exceeds the context window).

<!-- tab: Fine-tune -->
## Step 1: Run the fine-tune command

Every flag in the training command has a specific purpose. Understanding them lets you tune for your hardware and dataset.

```bash
mlx_lm.lora \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \   # base model — 4-bit quantized to fit in memory
  --data ./data \                                      # directory containing train.jsonl and valid.jsonl
  --train \                                            # enable training mode (vs inference-only)
  --iters 1000 \                                       # total gradient update steps
  --learning-rate 1e-5 \                               # conservative LR — safe for most datasets
  --batch-size 4 \                                     # examples per gradient step (reduce to 2 if OOM)
  --lora-layers 16 \                                   # add LoRA to the last 16 attention layers
  --lora-rank 8 \                                      # rank of adapter matrices (capacity)
  --val-batches 25 \                                   # validation steps between checkpoints
  --steps-per-report 50 \                              # print loss every 50 steps
  --steps-per-eval 200 \                               # run validation every 200 steps
  --save-every 200 \                                   # save checkpoint every 200 steps
  --adapter-path ./adapters                            # where to save adapter weights
```

The training output will look like:

```
Iter 50: Train loss 2.847, Learning Rate 1.000e-05, It/sec 1.23
Iter 100: Train loss 2.541, Learning Rate 1.000e-05, It/sec 1.24
Iter 200: Train loss 2.102, Val loss 2.198, Val took 12.3s
Iter 400: Train loss 1.876, Val loss 1.943, Val took 12.1s
```

## Step 2: Monitor training loss

The loss values tell you whether training is healthy. Knowing what to look for saves you from completing a 1000-step run that produced a useless adapter.

A healthy training curve:
- Train loss should decrease steadily from the initial value (~2.5–3.5 for a chat model)
- Val loss should track train loss closely — both declining
- After ~70% of training, loss plateaus (that's expected and correct)

Warning signs:
- **Val loss rising while train loss falls** — overfitting. Stop early and use the checkpoint from before the divergence.
- **Loss not moving after 200 steps** — learning rate too low, or data format mismatch. Check your JSONL format.
- **Loss suddenly spikes to NaN** — learning rate too high, or gradient explosion. Restart with `--learning-rate 5e-6`.
- **Loss stuck at same value** — batch_size too large for your dataset, or data all identical. Check for duplicate examples.

## Step 3: Test the adapter mid-training

Checkpoints are saved every `--save-every` steps. You can test the adapter quality at any checkpoint without waiting for training to complete.

```bash
# Test the adapter at the latest checkpoint
mlx_lm.generate \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \   # same base model used in training
  --adapter-path ./adapters \                          # path to adapter directory
  --prompt "<|im_start|>user\nWhat is the boiling point of water?<|im_end|>\n<|im_start|>assistant\n" \
  --max-tokens 100

# Compare against the base model without the adapter
mlx_lm.generate \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \   # no adapter-path
  --prompt "<|im_start|>user\nWhat is the boiling point of water?<|im_end|>\n<|im_start|>assistant\n" \
  --max-tokens 100
```

If the adapter is working, the fine-tuned output should show the style, format, or knowledge you trained on. If both outputs look identical, the adapter may not be learning — check your data format.

## Step 4: Resume interrupted training

If your training run is interrupted (power loss, accidental Ctrl+C, Mac sleep), resume from the last checkpoint rather than starting over.

```bash
# Resume from the last saved checkpoint
mlx_lm.lora \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --data ./data \
  --train \
  --resume-adapter-file ./adapters/adapters.npz \     # load existing adapter weights
  --iters 1000 \                                       # total iters — MLX continues from where it left off
  --learning-rate 1e-5 \
  --batch-size 4 \
  --lora-layers 16 \
  --adapter-path ./adapters                            # same output path
```

Checkpoints are named `adapters_<step>.npz`. Use the highest-numbered one unless validation showed that an earlier checkpoint had better val loss.

<!-- tab: Export -->
## Step 1: Merge the adapter into the model

`mlx_lm.fuse` permanently bakes the adapter weights into a copy of the base model. The resulting model has no separate adapter — the LoRA deltas (B·A) are added directly to each weight matrix W, producing a standard model with no inference overhead.

When to fuse vs keep separate:
- **Fuse** when you want a standalone model for deployment or sharing, or when converting to GGUF for Ollama
- **Keep separate** when you're experimenting and may train further, or when you need to swap between different adapters on the same base model

```bash
mlx_lm.fuse \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \   # base model
  --adapter-path ./adapters \                          # trained adapter
  --save-path ./fused-model                            # output directory for merged model
  --de-quantize                                        # optional: convert back to FP16 for GGUF export
```

Use `--de-quantize` only if you're converting to GGUF next — it produces a larger but more compatible output. Skip it if you're deploying with mlx_lm directly.

## Step 2: Test the merged model

Run the same prompt on the fused model to confirm the merge didn't introduce any artifacts. The output should be identical to running the base model + adapter separately.

```bash
mlx_lm.generate \
  --model ./fused-model \
  --prompt "<|im_start|>user\nTest your fine-tuned behavior here.<|im_end|>\n<|im_start|>assistant\n" \
  --max-tokens 200
```

## Step 3: Convert to GGUF for Ollama

To use your fine-tuned model with Ollama, convert it to GGUF format. This requires llama.cpp's conversion script.

```bash
# Step 1: Ensure the fused model is in FP16 (re-fuse with --de-quantize if needed)
mlx_lm.fuse \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --adapter-path ./adapters \
  --save-path ./fused-model-fp16 \
  --de-quantize                                        # produces FP16 weights for GGUF conversion

# Step 2: Install llama.cpp conversion dependencies
pip install gguf transformers                          # required by convert_hf_to_gguf.py

# Step 3: Convert to GGUF (requires llama.cpp source)
# Get llama.cpp if you don't have it: git clone https://github.com/ggml-org/llama.cpp
python llama.cpp/convert_hf_to_gguf.py \
  ./fused-model-fp16 \                                 # input: HF-format model directory
  --outtype q4_k_m \                                   # output quantization (q4_k_m recommended)
  --outfile ./my-fine-tuned-model.gguf                 # output file path

# Step 4: Create an Ollama Modelfile
cat > Modelfile << 'EOF'
FROM ./my-fine-tuned-model.gguf
TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
PARAMETER stop "<|im_end|>"
EOF

# Step 5: Register with Ollama
ollama create my-fine-tuned-qwen -f ./Modelfile

# Step 6: Test with Ollama
ollama run my-fine-tuned-qwen "Test your fine-tuned behavior here."
```

## Step 4: Share on Hugging Face (optional)

```bash
# Upload the adapter (small — ~50 MB for rank 8 on 7B model)
huggingface-cli upload your-username/qwen2.5-7b-my-adapter ./adapters

# Or upload the full fused model
huggingface-cli upload your-username/qwen2.5-7b-my-finetuned ./fused-model
```

<!-- tab: Troubleshooting -->
## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| OOM crash during training | batch_size too large for available RAM | Reduce `--batch-size` to 2 or 1; also try `--lora-layers 8` |
| Loss not decreasing after 300 steps | LR too low, or data format mismatch | Check data format with validation run; try `--learning-rate 5e-5` |
| Loss decreases then suddenly spikes | Learning rate too high | Restart with `--learning-rate 5e-6`; use the last good checkpoint |
| Adapter output identical to base model | Adapter not loaded, or trained on wrong data | Verify `--adapter-path` exists and contains `adapters.npz`; test with a known training example |
| Tokenizer errors — garbled output | Chat template mismatch | Your JSONL template doesn't match the model; re-check tokenizer_config.json |
| Training stalled at step 1 for > 5 min | Metal kernel JIT compilation | Normal for first run — Metal compiles GPU kernels on first use. Wait up to 10 min. |
| NaN loss from step 1 | Malformed data or LR too high | Validate your JSONL with `--iters 1`; lower LR to `1e-6` |
| Checkpoint `adapters.npz` not found on resume | Wrong path or training never saved | Check `ls ./adapters/` for `adapters_<N>.npz` files; use the highest N |

### Diagnosing overfitting with validation curves

Overfitting means the model is memorizing training examples rather than learning generalizable patterns. The diagnostic is simple: compare train loss and val loss over time.

```
Step 200:  train=2.10, val=2.15  ← Healthy: val slightly above train
Step 400:  train=1.87, val=1.92  ← Healthy: both declining together
Step 600:  train=1.71, val=1.98  ← Warning: val rising, train still falling
Step 800:  train=1.62, val=2.31  ← Overfit: stop and use step 400 checkpoint
```

When you see val loss rising, stop training and use the checkpoint from before the divergence:

```bash
# Use the step-400 checkpoint instead of the latest
mlx_lm.generate \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --adapter-path ./adapters/adapters_400.npz \    # specific checkpoint by step number
  --prompt "Your test prompt"
```

Prevention: reduce training data repetition, lower `--lora-rank`, or reduce `--lora-layers`.

### Choosing the right learning rate

The learning rate is the most sensitive hyperparameter. Use this heuristic:

```
Dataset size          Recommended LR range
< 200 examples        5e-6 to 2e-5   (very conservative — small datasets overfit quickly)
200–1000 examples     1e-5 to 5e-5   (standard range)
1000–5000 examples    5e-5 to 1e-4   (can afford faster learning)
> 5000 examples       1e-4 to 2e-4   (large dataset, can use higher LR safely)
```

If you're unsure, use `1e-5` — it's safe for all dataset sizes, just slower for large ones.

### Fixing "nan loss" from step 1

NaN (not a number) loss on the first step indicates a data or configuration problem, not a training dynamics problem:

```bash
# Check for empty lines or malformed JSON in your data
python3 -c "
import json
with open('data/train.jsonl') as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            print(f'Line {i}: EMPTY')
            continue
        try:
            obj = json.loads(line)
            if 'text' not in obj:
                print(f'Line {i}: Missing text field')
            elif len(obj[\"text\"]) == 0:
                print(f'Line {i}: Empty text field')
        except json.JSONDecodeError as e:
            print(f'Line {i}: JSON error: {e}')
print('Done checking')
"

# If data looks fine, try drastically lower LR
mlx_lm.lora --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --data ./data --train --iters 50 \
  --learning-rate 1e-6 \              # 10× lower than default
  --batch-size 1 \                    # minimum batch size
  --lora-layers 4 \                   # fewer layers = more stable
  --adapter-path ./adapters-debug
```

### Understanding batch size vs gradient accumulation on Apple Silicon

On Apple Silicon, Metal doesn't support gradient accumulation natively in MLX. This means `--batch-size` directly controls memory usage — there's no way to simulate a larger effective batch with accumulation.

Practical limits by machine:

| Machine RAM | Max batch_size for 7B Q4 | Max batch_size for 13B Q4 |
|---|---|---|
| 16 GB | 2 | 1 |
| 32 GB | 4–6 | 2 |
| 64 GB | 8–12 | 4–6 |
| 96 GB | 16 | 8 |

If OOM at batch_size=1, reduce `--lora-layers` first (try 8, then 4) before reducing the model size.
