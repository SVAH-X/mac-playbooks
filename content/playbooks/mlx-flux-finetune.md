---
slug: mlx-flux-finetune
title: "FLUX LoRA Fine-tuning"
time: "1 hr"
color: red
desc: "Fine-tune FLUX.1 image models with LoRA on Mac"
tags: [image generation, fine-tuning]
spark: "FLUX.1 Dreambooth LoRA"
category: fine-tuning
featured: false
whatsNew: false
---

<!-- tab: Overview -->
## Basic idea

FLUX.1 is a state-of-the-art diffusion transformer model for text-to-image generation, developed by Black Forest Labs. LoRA fine-tuning teaches it a new subject — your dog, a specific illustration style, a product — by showing it 10-30 example images paired with text descriptions. The model adjusts its internal representations to associate your specific subject with a trigger word you choose.

The result is a small adapter file (~50-200 MB) that you apply on top of the frozen FLUX.1 base model at generation time. mflux implements FLUX.1 LoRA training natively on Apple Silicon using Metal acceleration, with no CUDA required.

## What you'll accomplish

A trained LoRA adapter (~100 MB `.safetensors` file) that generates images of your specific subject when prompted with your trigger word. For example: `"a photo of sks_dog playing in the snow"` produces images of your dog rather than a generic dog. The adapter is usable with `mflux-generate` and compatible with other FLUX.1 tools.

## What to know before starting

- **Diffusion models**: FLUX.1 generates images by starting from random noise and iteratively denoising it toward an image described by the text prompt. The "model" is a learned function that predicts how to remove noise at each step.
- **What LoRA adds to image models**: LoRA injects a style/subject bias into the attention layers that are active during the denoising process. It biases the denoising trajectory toward your subject's appearance without retraining the entire 12 billion parameter model.
- **Trigger words**: The trigger word (`sks_dog`, `ohwx_cat`, etc.) disambiguates your subject from the general concept. Without it, the model has no way to distinguish "your dog" from "any dog." The word should be unusual — common words like "dog" confuse the model.
- **FLUX.1 dev vs schnell**: `dev` was trained for quality and requires 20-50 steps per image. `schnell` was distilled to 4 steps — faster, but lower quality and worse at learning new subjects. Use `dev` for LoRA fine-tuning.
- **Training data quality is the bottleneck**: Unlike LLM fine-tuning where more data is almost always better, image LoRA quality is limited by the consistency and quality of your input images. 15 excellent images outperform 50 mediocre ones.

## Prerequisites

- macOS 14.0 or later
- Apple Silicon Mac (M1, M2, or M3 family)
- Python 3.10 or later
- 32 GB+ unified memory (FLUX.1 dev weights are ~34 GB)
- 20-50 GB free disk space (model weights + checkpoints)
- 10-30 training images of your subject

## Time & risk

- **Duration**: ~1 hour setup and data prep, several hours of training
- **Risk level**: Medium — the initial FLUX.1 dev model download is ~34 GB and requires a HuggingFace account. Training is memory-intensive.
- **Rollback**: Delete the mflux pip install and the HuggingFace cache at `~/.cache/huggingface/hub`.

<!-- tab: Setup -->
## Step 1: Install mflux

mflux is the Apple Silicon-native FLUX.1 implementation. LoRA training support was added in version 0.4.0 — verify you have a recent enough version.

```bash
# Install or upgrade mflux
pip install --upgrade mflux

# Verify the version includes LoRA training (need >= 0.4.0)
python -c "import mflux; print(mflux.__version__)"

# Verify the train command is available
mflux-train --help | head -20
```

## Step 2: Prepare training images

Image quality determines LoRA quality more than any hyperparameter. Use these criteria:

- **Sharpness**: No motion blur, no soft focus. The model needs to learn fine details.
- **Subject prominence**: Your subject should fill at least 40% of the frame.
- **Lighting variety**: Front lit, side lit, natural light, indoor light — variety prevents the LoRA from learning the lighting as part of the subject.
- **Background variety**: Different backgrounds prevent the LoRA from memorizing the background along with the subject.
- **No text or watermarks**: These become part of what the LoRA learns.
- **Resolution**: 512x512 minimum, 1024x1024 ideal. Mismatched aspect ratios are fine — mflux crops during training.

```bash
# Create a directory with a descriptive name matching your trigger word
mkdir -p ~/lora-training/sks_dog

# Place your images here — any of these formats work
ls ~/lora-training/sks_dog/
# img_001.jpg  img_002.jpg  img_003.png  ...

# Count your images — aim for 15-25
ls ~/lora-training/sks_dog/ | wc -l
```

iPhone photos work well — they're high resolution, sharp, and consistent. Take photos in various locations and lighting conditions on the same day for maximum variety.

## Step 3: Test the base FLUX model before training

Before spending hours training, confirm mflux generates images correctly on your machine. This also gives you a baseline to compare against after training.

```bash
# Generate a baseline image of a subject similar to yours
# This downloads FLUX.1 dev weights (~34GB) on first run
mflux-generate \
  --model dev \
  --prompt "a golden retriever dog playing in the snow, photorealistic" \
  --steps 20 \
  --seed 42 \
  --width 1024 --height 1024 \
  --output baseline_before_training.png
```

The first run downloads the model weights — this takes 20-40 minutes depending on your internet connection. Subsequent runs are fast since the weights are cached at `~/.cache/huggingface/hub`.

<!-- tab: Train -->
## Step 1: Start training

Every flag affects training behavior and quality. Understand them before tuning:

```bash
mflux-train \
  --model dev \                        # Use dev for quality; schnell doesn't LoRA train well
  --train-data-dir ~/lora-training/sks_dog \  # Directory containing your images
  --trigger-word "sks_dog" \           # Unique token — use underscore, not space
  --steps 1000 \                       # 500 = quick test; 1000-2000 = production quality
  --output-dir ~/lora-output/sks_dog \ # Where adapter checkpoints are saved
  --learning-rate 1e-4 \               # Conservative default; lower = slower but safer
  --rank 8 \                           # LoRA rank: higher = more capacity, more memory
  --save-checkpoint-every 200          # Save intermediate checkpoints for comparison
```

The `--trigger-word` value becomes part of all training captions automatically. Choose something that couldn't appear by accident in a normal prompt: `sks_dog`, `ohwx_style`, `xkq_product` — the pattern `xxx_noun` works well.

## Step 2: Monitor training progress

The training loop prints loss at each step:

```
Step 100/1000  loss: 0.1832  lr: 0.000100
Step 200/1000  loss: 0.1421  lr: 0.000100
Step 300/1000  loss: 0.1203  lr: 0.000100
```

Unlike LLM training, **very low loss means overfitting** in image LoRA training. The model has memorized your training images and will reproduce them with minimal variation. The goal is a loss plateau — not the minimum. Healthy patterns:

- Loss decreasing steadily: training is working
- Loss plateau (~0.08-0.15 for typical subjects): approaching the sweet spot
- Loss near 0 (<0.02): overfitting — stop training, go back to an earlier checkpoint

## Step 3: Test at checkpoint intervals

The best way to find the sweet spot is to generate test images every 200 steps and compare:

```bash
# Generate a test image at each checkpoint
for step in 200 400 600 800 1000; do
  checkpoint="~/lora-output/sks_dog/adapter_${step}.safetensors"
  if [ -f "$checkpoint" ]; then
    mflux-generate \
      --model dev \
      --prompt "a photo of sks_dog sitting in a park, photorealistic" \
      --lora-path "$checkpoint" \
      --lora-scale 0.9 \
      --steps 20 \
      --seed 42 \
      --output "test_step_${step}.png"
    echo "Generated test image for step $step"
  fi
done
```

Open the generated images side by side. The best checkpoint is when the subject is recognizable and the image has creative variety.

## Step 4: Decide when to stop

Stop training when generated images:
- Clearly resemble your training subject
- Still show variety across different prompts (not just copies of training images)
- Respond to prompt changes (e.g., "in the snow" vs "in a park" produces different backgrounds)

If your images look creative but don't resemble the subject, train longer (more steps or higher learning rate). If your images always look like copies of your training photos regardless of prompt, you've overfit — roll back to an earlier checkpoint.

<!-- tab: Generate -->
## Step 1: Generate images with your trained LoRA

The `--lora-scale` parameter controls how strongly the LoRA influences generation. 1.0 = full strength, 0.0 = no LoRA (base model behavior). Start at 0.9 and adjust:

```bash
mflux-generate \
  --model dev \
  --prompt "a photo of sks_dog sitting by a fireplace, cozy, warm lighting" \
  --lora-path ~/lora-output/sks_dog/adapter_final.safetensors \
  --lora-scale 0.9 \           # Start here — tune between 0.7 and 1.1
  --steps 20 \
  --seed 42 \
  --width 1024 --height 1024 \
  --output output.png
```

## Step 2: Prompt engineering with trigger words

The trigger word must appear in your prompt exactly as specified during training. Effective prompt structures:

```bash
# Pattern: "a [medium] of [trigger_word] [action], [style modifiers]"

# Photorealistic style
mflux-generate --model dev \
  --prompt "a photo of sks_dog running on a beach, golden hour, photorealistic, 8k" \
  --lora-path ~/lora-output/sks_dog/adapter_final.safetensors \
  --lora-scale 0.9 --steps 20 --seed 100 --output beach.png

# Artistic style
mflux-generate --model dev \
  --prompt "an oil painting of sks_dog, impressionist style, vibrant colors" \
  --lora-path ~/lora-output/sks_dog/adapter_final.safetensors \
  --lora-scale 0.85 --steps 25 --seed 200 --output painting.png

# Studio portrait
mflux-generate --model dev \
  --prompt "a studio portrait of sks_dog, white background, professional photography" \
  --lora-path ~/lora-output/sks_dog/adapter_final.safetensors \
  --lora-scale 0.95 --steps 20 --seed 300 --output portrait.png
```

## Step 3: Find the optimal LoRA scale

Generate a comparison grid at different LoRA scales to find what looks best for your subject:

```bash
# Compare LoRA scale values for the same prompt and seed
for scale in 0.5 0.7 0.9 1.0 1.1; do
  mflux-generate \
    --model dev \
    --prompt "a photo of sks_dog in a field of flowers" \
    --lora-path ~/lora-output/sks_dog/adapter_final.safetensors \
    --lora-scale $scale \
    --steps 20 \
    --seed 42 \                  # Same seed = same noise, only LoRA changes
    --output "scale_${scale}.png"
  echo "Generated scale=$scale"
done

# View all outputs together to compare
open scale_*.png
```

## Step 4: Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: MPS out of memory` during training | FLUX.1 dev too large | Use `--rank 4` (lower memory) or quantize with `--quantize 8` |
| Download fails with 403 error | FLUX.1 dev is a gated model | Run `huggingface-cli login` with a token that has accepted the FLUX.1 license |
| Loss drops to near 0 in first 50 steps | Training data too small or LR too high | Use 15+ images and reduce `--learning-rate` to `5e-5` |
| Generated images look nothing like subject | Trigger word missing from prompt | Include `sks_dog` (or your trigger word) exactly in the prompt |
| Generated images look identical to training photos | Overfitting | Use an earlier checkpoint (`adapter_500.safetensors` instead of final) |
| `mflux-train: command not found` | mflux version < 0.4.0 | Run `pip install --upgrade mflux` to get a version with training support |
