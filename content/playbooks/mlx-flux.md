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

FLUX.1 is a state-of-the-art image generation model developed by Black Forest Labs. Unlike Stable Diffusion (which uses a U-Net denoiser), FLUX.1 is a diffusion transformer (DiT) trained with a flow-matching objective — a newer technique that produces straighter paths through the noise manifold, yielding better image quality and prompt adherence. `mflux` is a pure MLX implementation that runs FLUX.1 natively on Apple Silicon, bypassing PyTorch entirely. It uses Metal (Apple's GPU compute API) through MLX's unified memory model, achieving 30–60 seconds per 1024×1024 image on M2 Max at 20 steps, or 8–15 seconds with the schnell (4-step) variant.

## What you'll accomplish

A working local image generation setup that produces 1024×1024 images from text prompts using both FLUX.1 schnell (fast, 4 steps, no HuggingFace token needed) and FLUX.1 dev (higher quality, 20 steps). Images are saved as PNG files locally. You will understand what each command-line flag controls and be able to tune generation for your hardware.

## What to know before starting

- **Diffusion models** — Generation works by starting from Gaussian noise (a random tensor), then taking N denoising steps. Each step runs the model to predict and subtract a bit of noise. After N steps you have a coherent image. More steps = higher quality but slower.
- **Flow matching** — FLUX's training objective. Instead of DDPM's curved noise paths, flow matching learns straighter "flows" from noise to image, meaning fewer steps are needed for good quality. This is why schnell works in 4 steps.
- **schnell vs dev** — `schnell` (German: fast) is distilled via consistency distillation to 4 steps. It's the quick, open-weight model. `dev` is the full-quality model (20–50 steps), gated on HuggingFace — you must accept its license before downloading.
- **Quantization** — FLUX.1 in full float16 requires ~34 GB of memory. 8-bit quantization reduces this to ~17 GB; 4-bit to ~9 GB. Quality degrades slightly with 4-bit but remains excellent for most prompts.
- **CFG (classifier-free guidance)** — Controls how strongly the model follows your prompt vs generating freely. schnell does not use CFG (it's baked into distillation). dev uses a `guidance_scale` parameter; 3.5–7.0 is the useful range.

## Prerequisites

- macOS 14.0+ (Sonoma) — required for latest MLX Metal kernels
- Apple Silicon Mac (M1, M2, M3, or M4 series)
- Python 3.10+
- 16 GB+ unified memory (minimum for schnell 4-bit); 32 GB+ recommended for dev 4-bit
- HuggingFace account (only required for FLUX.1 dev)

## Time & risk

- **Duration:** 15 minutes setup; first run downloads ~9 GB (schnell 4-bit) or ~17 GB (dev 8-bit)
- **Risk level:** Low — no system changes, only pip package and model download
- **Rollback:** `pip uninstall mflux`; delete `~/.cache/huggingface/` to reclaim disk space

<!-- tab: Install -->
## Step 1: Install mflux

`mflux` installs MLX and all necessary transformer/tokenizer dependencies automatically. No PyTorch or CUDA required.

```bash
pip install mflux
# This also installs: mlx, transformers, huggingface-hub, Pillow, tqdm

# Verify installation and check version
python -c "import mflux; print(mflux.__version__)"
# Expected: 0.4.0 or newer

# Verify MLX can see your GPU
python -c "import mlx.core as mx; print(mx.default_device())"
# Expected: Device(gpu, 0) — if you see Device(cpu, 0), MLX is not using Metal
```

## Step 2: HuggingFace login (for FLUX.1 dev only)

FLUX.1 dev is a gated model — Black Forest Labs requires you to accept the license before downloading. schnell is open and requires no login.

```bash
# Install the HuggingFace CLI
pip install huggingface_hub

# Log in with your HuggingFace token (get it at huggingface.co/settings/tokens)
huggingface-cli login
# Paste your token when prompted — it is stored in ~/.huggingface/token

# Before running: visit https://huggingface.co/black-forest-labs/FLUX.1-dev
# and click "Access repository" to accept the license
```

## Step 3: Test with FLUX.1 schnell

The first run downloads and caches the model. This takes several minutes depending on your internet speed. Subsequent runs start immediately.

```bash
mflux-generate \
  --model schnell \
  --prompt "A red apple on a white table, studio lighting, product photography" \
  --steps 4 \
  --seed 42 \
  --width 1024 --height 1024
# First run: downloads ~9 GB to ~/.cache/huggingface/hub/
# Output: saved as mflux_generated_[timestamp].png in current directory
# Generation time: 8-15 seconds on M2/M3, 20-40 seconds on M1
```

## Step 4: Verify output

Open the generated PNG and check that it looks like your prompt. A good first test uses a concrete, descriptive prompt — FLUX is better at literal descriptions than abstract concepts.

```bash
# Open the generated image
open mflux_generated_*.png

# Check file size — a 1024x1024 PNG should be 1-4 MB
ls -lh mflux_generated_*.png

# If the image is black or blank, check Step 3 in Troubleshooting below
```

<!-- tab: Generate -->
## Step 1: schnell basic generation

`schnell` is the fast model. Use it for iteration, prompt testing, and any task where you'll generate many images.

```bash
mflux-generate \
  --model schnell \
  --prompt "A detailed watercolor painting of a lighthouse at dusk, pastel colors, reflections in calm water" \
  --steps 4 \
  --seed 42 \
  --width 1024 \
  --height 1024 \
  --output lighthouse.png
# --model schnell: selects the 4-step distilled model
# --steps 4: do not reduce below 4 for schnell — quality degrades sharply
# --seed 42: fixes the initial noise tensor — same seed = same image every time
# --width/--height: output resolution in pixels (must be multiples of 64)
# --output: explicit filename instead of timestamp-based default
```

## Step 2: dev generation

`dev` produces higher-quality images, especially for complex scenes, accurate text rendering, and photorealism. It requires 20+ steps and a HuggingFace token.

```bash
mflux-generate \
  --model dev \
  --prompt "A photorealistic portrait of an astronaut on Mars, golden hour lighting, dust in the atmosphere, NASA equipment visible in background" \
  --steps 20 \
  --seed 100 \
  --guidance 3.5 \
  --width 1024 \
  --height 1024 \
  --output astronaut_mars.png
# --model dev: full-quality model, downloads ~17 GB on first run (8-bit default)
# --steps 20: minimum for dev quality; 30-50 for maximum fidelity
# --guidance 3.5: CFG scale — range 1.0-10.0; 3.5 is balanced; 7.0 is very prompt-adherent
```

## Step 3: Quantized generation to save memory

Quantization reduces the model's memory footprint by representing weights in lower precision. Use this if you hit memory limits or want faster generation.

```bash
# 4-bit quantization — ~9 GB, fastest, slight quality reduction
mflux-generate \
  --model schnell \
  --prompt "Abstract geometric art, bold primary colors, Mondrian style" \
  --steps 4 \
  --quantize 4 \
  --seed 7

# 8-bit quantization — ~17 GB, good quality, moderate speed
mflux-generate \
  --model dev \
  --prompt "Abstract geometric art, bold primary colors, Mondrian style" \
  --steps 20 \
  --quantize 8 \
  --guidance 4.0

# Memory usage reference:
# Model       | No quant | 8-bit | 4-bit
# schnell     |  ~34 GB  | ~17GB | ~9 GB
# dev         |  ~34 GB  | ~17GB | ~9 GB
```

## Step 4: Aspect ratios and resolution

FLUX supports any resolution that is a multiple of 64. Total pixel count affects generation time more than either dimension individually.

```bash
# Portrait (9:16 — social media vertical)
mflux-generate --model schnell --prompt "Portrait of a fox in a forest" \
  --steps 4 --seed 1 --width 768 --height 1344

# Landscape (16:9 — widescreen)
mflux-generate --model schnell --prompt "Panoramic view of a mountain range at sunrise" \
  --steps 4 --seed 1 --width 1344 --height 768

# Square (1:1 — default)
mflux-generate --model schnell --prompt "Close-up of a sunflower, macro photography" \
  --steps 4 --seed 1 --width 1024 --height 1024

# Note: total megapixels = width * height / 1,000,000
# 1024x1024 = 1.05 MP; 1344x768 = 1.03 MP — similar generation time
# Going to 1536x1536 (2.36 MP) roughly doubles generation time
```

<!-- tab: Tips -->
## Prompt engineering for FLUX

FLUX.1 follows prompts more literally than Stable Diffusion. It responds well to specific, descriptive language that describes visual elements — not mood or feeling.

```
Good: "A red ceramic mug on a dark wood table, steam rising, morning light from left window, shallow depth of field"
Weak: "A cozy morning coffee scene with warm vibes"

Good: "The word 'HELLO' written in large blue neon letters on a black background"
Note: FLUX.1 is unusually good at text rendering — describe text exactly as you want it
```

## Batch generation script

Generate multiple images with different seeds to explore variations on the same prompt.

```bash
#!/bin/bash
PROMPT="A futuristic city at night, neon lights reflecting in rain puddles, cyberpunk aesthetic"

for SEED in 1 2 3 4 5; do
  mflux-generate \
    --model schnell \
    --prompt "$PROMPT" \
    --steps 4 \
    --seed "$SEED" \
    --width 1024 \
    --height 1024 \
    --output "city_seed_${SEED}.png"
  echo "Generated seed $SEED"
done
# Compare the 5 images and pick the best composition before doing a high-quality dev run
```

## Python API for programmatic generation

Use the Python API when you want to integrate generation into a script or notebook.

```python
from mflux import Flux1, Config

# Load model once (expensive) — reuse for multiple generations
flux = Flux1.from_alias(
    alias="schnell",    # or "dev"
    quantize=4,         # 4-bit quantization
)

# Generate image
image = flux.generate_image(
    seed=42,
    prompt="An oil painting of a cat wearing a Victorian-era top hat and monocle",
    config=Config(
        num_inference_steps=4,   # 4 for schnell, 20+ for dev
        height=1024,
        width=1024,
        # guidance=3.5,          # uncomment for dev model only
    )
)
image.save("cat_portrait.png")
print("Saved cat_portrait.png")
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Download fails midway | Disk space or network interruption | Check `df -h` for free space; re-run the command — partial downloads resume |
| `401 Unauthorized` during download | HuggingFace token missing or expired | Run `huggingface-cli login` again; re-accept the dev model license at HF website |
| Out of memory (OOM) error | Model too large for available RAM | Add `--quantize 4` for schnell; on 16 GB Macs use schnell only (dev needs 32 GB) |
| Generated image is solid black or gray | Model partially downloaded / corrupted | Delete cache: `rm -rf ~/.cache/huggingface/hub/models--black-forest-labs*` and re-download |
| Very slow generation (> 5 min for schnell) | MLX is using CPU not Metal GPU | Verify with `python -c "import mlx.core as mx; print(mx.default_device())"` — must show `gpu` |
| Blurry or low-detail output | Too few steps for dev model | Use `--steps 20` minimum for dev; schnell is designed for exactly 4 steps |
| Repeated tiling pattern in image | Extremely high guidance scale | Lower `--guidance` to 3.5–5.0 range; values above 8.0 cause artifacts |
| `model not found` after clearing disk | HuggingFace cache was deleted | Re-run generation command — model will re-download automatically |

### OOM error on 16 GB Macs

On 16 GB unified memory Macs, the system uses ~6–8 GB for macOS, leaving ~8–10 GB for applications. FLUX.1 schnell with 4-bit quantization needs ~9 GB — right at the limit. Tips:

```bash
# Close all other applications before generating
# Use 4-bit quantization
mflux-generate --model schnell --quantize 4 --steps 4 \
  --prompt "your prompt" --width 1024 --height 1024

# If still OOM, reduce resolution
mflux-generate --model schnell --quantize 4 --steps 4 \
  --prompt "your prompt" --width 768 --height 768
# 768x768 uses roughly half the memory of 1024x1024
```

### Verifying Metal GPU usage

If generation is slow, confirm MLX is using the GPU:

```bash
# While generating, open a second terminal and run:
sudo powermetrics --samplers gpu_power -i 1000 -n 5
# You should see GPU Active Residency > 80% during generation
# If GPU is at 0%, MLX fell back to CPU — reinstall mlx: pip install --upgrade mlx
```
