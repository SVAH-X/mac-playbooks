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

Fine-tune FLUX.1 image generation models with LoRA to generate images in a specific style or of a specific subject. Runs natively on Apple Silicon via MLX.

## Prerequisites

- macOS 14.0+
- Apple Silicon Mac
- Python 3.10+
- 32 GB+ unified memory (64 GB recommended for FLUX.1 dev)

## Time & risk

- **Duration:** 1 hour setup, several hours training
- **Risk level:** Medium — large memory requirements

<!-- tab: Setup -->
## Install mflux

```bash
pip install mflux
```

## Prepare training images

Collect 10–30 high-quality images of your subject. Recommended:
- 512×512 or 1024×1024 resolution
- Consistent subject framing
- Varied backgrounds and lighting

Place images in a directory:
```
training_images/
├── img_001.jpg
├── img_002.jpg
└── ...
```

<!-- tab: Train -->
## Start LoRA training

```bash
mflux-train \
  --model dev \
  --train-data-dir ./training_images \
  --trigger-word "sks" \
  --steps 500 \
  --output-dir ./lora-output
```

## Training parameters

- `--steps 500` — number of training steps (increase for better quality)
- `--trigger-word "sks"` — unique token to trigger your style/subject
- `--learning-rate 1e-4` — default, adjust if needed

<!-- tab: Generate -->
## Generate with trained LoRA

```bash
mflux-generate \
  --model dev \
  --prompt "A photo of sks dog in a park" \
  --lora-path ./lora-output/adapter.safetensors \
  --steps 20 \
  --seed 42 \
  --width 1024 --height 1024
```

## Test with increasing LoRA strength

```bash
for weight in 0.5 0.75 1.0 1.25; do
  mflux-generate \
    --model dev \
    --prompt "A painting of sks dog" \
    --lora-path ./lora-output/adapter.safetensors \
    --lora-scale $weight \
    --steps 20 --seed 42
done
```
