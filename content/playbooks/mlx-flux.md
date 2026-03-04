---
slug: mlx-flux
title: "FLUX / Stable Diffusion with MLX"
time: "15 min"
color: green
desc: "Generate images with FLUX and SD models natively on Mac"
tags: [image generation, mlx]
spark: "FLUX.1 Dreambooth LoRA"
category: image-gen
featured: true
whatsNew: false
---

<!-- tab: Overview -->
## Basic idea

Generate high-quality images with FLUX.1 and Stable Diffusion models entirely on your Mac using MLX and mflux. No cloud API needed.

## Prerequisites

- macOS 14.0+
- Apple Silicon Mac
- Python 3.10+
- 16 GB+ unified memory (32 GB recommended for FLUX.1 dev)

## Time & risk

- **Duration:** 15 minutes setup
- **Risk level:** Low
- **Note:** First run downloads models (~12 GB for FLUX.1 schnell)

<!-- tab: Install -->
## Install mflux (FLUX with MLX)

```bash
pip install mflux
```

## Or install mlx-image for Stable Diffusion

```bash
pip install mlx-image
```

<!-- tab: Generate -->
## Generate with FLUX.1 schnell (fast, 4 steps)

```bash
mflux-generate \
  --model schnell \
  --prompt "A serene Japanese garden with cherry blossoms, digital art" \
  --steps 4 \
  --seed 42 \
  --width 1024 --height 1024
```

## Generate with FLUX.1 dev (higher quality, 20 steps)

```bash
mflux-generate \
  --model dev \
  --prompt "A futuristic cityscape at sunset, photorealistic, cinematic lighting" \
  --steps 20 \
  --seed 42 \
  --width 1024 --height 1024
```

## Python API

```python
from mflux import Flux1, Config

flux = Flux1.from_alias("schnell")
image = flux.generate_image(
    seed=42,
    prompt="An oil painting of a cat wearing a space suit",
    config=Config(num_inference_steps=4, height=1024, width=1024)
)
image.save("output.png")
```

<!-- tab: Tips -->
## Quantize for faster generation

```bash
# 4-bit quantization for speed
mflux-generate \
  --model schnell \
  --prompt "Your prompt here" \
  --quantize 4 \
  --steps 4
```

## Memory usage by model

| Model | Full precision | 8-bit | 4-bit |
|---|---|---|---|
| FLUX.1 schnell | ~34 GB | ~17 GB | ~9 GB |
| FLUX.1 dev | ~34 GB | ~17 GB | ~9 GB |
| SDXL | ~7 GB | ~3.5 GB | ~2 GB |

## Batch generation

```bash
for seed in 1 2 3 4 5; do
  mflux-generate \
    --model schnell \
    --prompt "A sunset over mountains" \
    --seed $seed \
    --steps 4 \
    --output "output_seed_${seed}.png"
done
```
