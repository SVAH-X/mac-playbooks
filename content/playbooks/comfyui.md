---
slug: comfyui
title: "ComfyUI on macOS"
time: "45 min"
color: orange
desc: "Node-based image generation workflow with MPS backend"
tags: [image generation, ui]
spark: "Comfy UI"
category: image-gen
featured: true
whatsNew: false
---

<!-- tab: Overview -->
## Basic idea

ComfyUI is a powerful node-based GUI for Stable Diffusion and FLUX workflows. It runs on macOS using the PyTorch MPS backend for GPU acceleration.

## Prerequisites

- macOS 12.3+
- Apple Silicon Mac (Intel Macs work but are significantly slower)
- Python 3.10+
- 16 GB+ unified memory recommended

## Time & risk

- **Duration:** 45 minutes (including model downloads)
- **Risk level:** Low

<!-- tab: Install -->
## Clone and install

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt
```

## Download a model

Place Stable Diffusion or FLUX checkpoint files in `models/checkpoints/`:

```bash
# Example: download SDXL base (6.9 GB)
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 \
  sd_xl_base_1.0.safetensors \
  --local-dir ./models/checkpoints
```

## Start ComfyUI

```bash
python main.py --force-fp16
```

Open http://127.0.0.1:8188 in your browser.

<!-- tab: First Workflow -->
## Load a basic text-to-image workflow

1. Open ComfyUI in your browser
2. The default graph shows a basic txt2img workflow
3. Double-click the **CheckpointLoaderSimple** node → select your model
4. Edit the positive prompt in the **CLIPTextEncode** node
5. Click **Queue Prompt** to generate

## Basic workflow components

- **CheckpointLoaderSimple** — loads your model
- **CLIPTextEncode** (positive) — your prompt
- **CLIPTextEncode** (negative) — what to avoid
- **KSampler** — the main sampling node (steps, CFG scale, sampler)
- **VAEDecode** — converts latents to image
- **SaveImage** — saves to `output/` folder

<!-- tab: Troubleshooting -->
## MPS out of memory

Reduce image resolution (512×512 instead of 1024×1024) or use a more quantized model.

## Custom nodes not working

Some custom nodes use CUDA operations and won't work on MPS. Check the node's GitHub for macOS compatibility.

## Slow generation

ComfyUI on MPS is slower than CUDA but faster than CPU. Using `--force-fp16` flag helps significantly.

## App crashes on startup

Ensure PyTorch is properly installed for your Python version:
```bash
pip install --upgrade torch torchvision torchaudio
```
