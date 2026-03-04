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

ComfyUI is a node-based visual programming environment for image generation with Stable Diffusion and FLUX models. Instead of a single prompt box, you build a directed acyclic graph (DAG) where each node does exactly one thing: load a model, encode text, sample latents, decode to pixels, save an image. This visual composition lets you build complex pipelines — inpainting, ControlNet guidance, LoRA stacking, img2img, upscaling — by wiring nodes together without writing code. On macOS, ComfyUI uses PyTorch's MPS (Metal Performance Shaders) backend to accelerate inference on Apple Silicon's GPU.

## What you'll accomplish

ComfyUI running at `localhost:8188` with a working text-to-image pipeline using SDXL, a clear understanding of how to build and modify node graphs, and the knowledge to install custom nodes for additional capabilities like ControlNet and upscalers.

## What to know before starting

- **Stable Diffusion architecture** — Three components work in sequence: (1) CLIP text encoder converts your prompt to a 77-token embedding, (2) UNet denoiser iteratively removes noise from a latent tensor guided by that embedding, (3) VAE decoder translates the latent tensor into a full-resolution RGB image.
- **Latent space** — The UNet doesn't work on pixels. It works on a compressed 4-channel latent representation that is 8× smaller in each spatial dimension (a 1024×1024 image is a 128×128×4 latent). This makes the denoising steps computationally tractable.
- **Sampler** — The algorithm that takes denoising steps. Different samplers (euler, dpmpp_2m, dpmpp_2m_karras) produce different quality/speed tradeoffs. Karras variants use a noise schedule that often improves results.
- **CFG scale** — Classifier-Free Guidance scale. Higher values make the output follow the prompt more strictly but can cause over-saturation and artifacts. For SDXL: 6–8 is a good range; above 12 often degrades quality.
- **VAE** — The Variational Autoencoder decoder. Each model checkpoint typically ships with a built-in VAE, but external VAE files (e.g., SDXL's `sdxl_vae.safetensors`) can improve color accuracy and sharpness.
- **safetensors format** — The secure model checkpoint format that ComfyUI expects. Avoids the arbitrary code execution risk of `.ckpt` (pickle) files. Always prefer `.safetensors` when downloading models.

## Prerequisites

- macOS 12.3+ (Monterey) — minimum for MPS backend in PyTorch
- Apple Silicon Mac (Intel Macs work but are significantly slower without MPS acceleration)
- Python 3.10+
- 16 GB+ unified memory
- 10–30 GB free disk space for models

## Time & risk

- **Duration:** 45 minutes including SDXL model download (6.9 GB)
- **Risk level:** Low — ComfyUI runs as a local web server, no system changes
- **Rollback:** Delete the `ComfyUI/` directory and uninstall Python packages

<!-- tab: Install -->
## Step 1: Clone ComfyUI

We clone the repository rather than installing via pip because ComfyUI is under active development and custom nodes expect the full source tree present on disk.

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
# This creates a ComfyUI/ directory with the web interface, node definitions,
# and the models/ directory structure that checkpoints must live in
ls models/
# Expected: checkpoints/  clip/  controlnet/  embeddings/  loras/  unet/  vae/
```

## Step 2: Install Python requirements

The `requirements.txt` includes PyTorch with MPS support, torchvision, and all ComfyUI node dependencies. The `--force-fp16` flag used at runtime (not here) converts float32 to float16 for MPS acceleration.

```bash
# Create a virtual environment to keep ComfyUI's dependencies isolated
python3 -m venv venv
source venv/bin/activate
# Your prompt should now show (venv)

# Install all dependencies (this takes 2-5 minutes)
pip install -r requirements.txt
# Key packages being installed:
# torch, torchvision, torchaudio — PyTorch with MPS backend
# transformers — for CLIP text encoders
# safetensors — for loading .safetensors checkpoint files
# Pillow, kornia — image processing
# aiohttp — ComfyUI's async web server

python -c "import torch; print(torch.backends.mps.is_available())"
# Must print True — if False, your PyTorch doesn't support MPS
```

## Step 3: Download a checkpoint model

ComfyUI loads models from `models/checkpoints/`. Download SDXL base — it produces 1024×1024 images natively and has excellent prompt adherence.

```bash
# Install huggingface-cli if not already present
pip install huggingface_hub

# Download SDXL base checkpoint (6.9 GB)
huggingface-cli download \
  stabilityai/stable-diffusion-xl-base-1.0 \
  sd_xl_base_1.0.safetensors \
  --local-dir ./models/checkpoints
# This saves sd_xl_base_1.0.safetensors in the checkpoints directory
# ComfyUI scans this directory at startup and shows all .safetensors files in dropdowns

ls -lh models/checkpoints/
# Expected: sd_xl_base_1.0.safetensors  6.9G
```

## Step 4: Start ComfyUI

The `--force-fp16` flag is critical on macOS MPS. It converts float32 operations to float16, reducing memory usage by 50% and improving generation speed significantly.

```bash
# Activate your virtual environment if not already active
source venv/bin/activate

# Start ComfyUI with MPS float16 optimization
python main.py --force-fp16
# Expected output:
# Total VRAM 0 MB, total RAM 32768 MB   (MPS uses unified memory, not separate VRAM)
# Starting server...
# To see the GUI go to: http://127.0.0.1:8188

# Open ComfyUI in your browser
open http://127.0.0.1:8188
# The node graph editor loads automatically
```

<!-- tab: First Workflow -->
## Step 1: Load the default workflow

When ComfyUI opens, it loads a default text-to-image workflow automatically. This graph has 6 nodes forming a minimal but complete txt2img pipeline.

The default nodes are:
- **CheckpointLoaderSimple** — loads your model file
- **CLIPTextEncode** (positive) — encodes your prompt
- **CLIPTextEncode** (negative) — encodes what to avoid
- **KSampler** — runs the denoising loop
- **VAEDecode** — converts latents to pixels
- **SaveImage** — writes the PNG to `ComfyUI/output/`

If the graph looks empty, click the menu icon (top-left) → Load Default.

## Step 2: Configure the CheckpointLoaderSimple

This node loads your model file from disk. Its output provides three things: the MODEL (UNet weights), CLIP (text encoder), and VAE (decoder) — all three flow to other nodes.

```
1. Click the CheckpointLoaderSimple node to select it
2. Click the "ckpt_name" dropdown
3. Select "sd_xl_base_1.0.safetensors" from the list
   — If the file doesn't appear, verify it's in models/checkpoints/ and restart ComfyUI
4. The node title updates to show the selected model name
```

The three output ports (MODEL, CLIP, VAE) are color-coded. Yellow cables carry the model weights; green cables carry CLIP encodings; red cables carry latent tensors.

## Step 3: Edit the CLIPTextEncode (positive prompt)

CLIP tokenizes text to a maximum of 77 tokens. Longer prompts are silently truncated. Write concise, specific descriptions.

```
1. Double-click the CLIPTextEncode node connected to the "positive" port of KSampler
2. Replace the example text with your prompt:
   "A majestic golden retriever in a sunlit forest, dappled light, professional photography, sharp focus, bokeh background"

Prompt syntax tips:
- Commas separate independent concepts: "red apple, white plate, studio lighting"
- Parentheses increase weight: "(sharp focus:1.3)" — 30% more emphasis
- Square brackets decrease weight: "[blurry:0.7]"
- Keep total under 60 words to stay within the 77-token limit
```

## Step 4: Configure the KSampler

The KSampler is the core node — it runs the iterative denoising process. Each setting meaningfully affects output quality.

```
seed:         42        — Initial noise state. Same seed + same settings = same image.
                          Click the dice icon to randomize for each generation.
control_after_generate: fixed  — "fixed" reuses seed; "randomize" picks a new seed each time
steps:        25        — Denoising iterations. SDXL: 20-30 is the sweet spot.
                          Below 15: often blurry. Above 40: diminishing returns.
cfg:          7.0       — Classifier-Free Guidance. SDXL: 6-8 recommended.
                          Higher = more literal but can over-saturate.
sampler_name: dpmpp_2m  — The denoising algorithm. "dpmpp_2m" with "karras" scheduler
                          gives excellent quality/speed on SDXL.
scheduler:    karras    — Noise schedule. Karras uses a specific sigma sequence that
                          front-loads the important high-noise steps.
denoise:      1.0       — 1.0 = full generation from noise. Use < 1.0 for img2img.
```

## Step 5: Queue the prompt and view output

```
1. Click "Queue Prompt" (bottom-right button) or press Ctrl+Enter
2. Watch the progress bar in the node graph — the KSampler shows step progress
3. When complete, the SaveImage node displays the generated image in the graph
4. Images are saved to: ComfyUI/output/ComfyUI_[number].png

Output metadata:
Every saved PNG contains the full workflow JSON embedded in its metadata.
Drag any ComfyUI-generated image back into the browser to reload its exact workflow.
This means you can share a PNG and the recipient can reproduce your exact settings.
```

<!-- tab: Troubleshooting -->
## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Generation takes 10+ min per image | `--force-fp16` not enabled or MPS fallback | Restart with `python main.py --force-fp16`; confirm `torch.backends.mps.is_available()` is True |
| Model not in CheckpointLoader dropdown | File in wrong directory or wrong extension | Move `.safetensors` file to `ComfyUI/models/checkpoints/`; restart ComfyUI |
| Black image output | VAE mismatch (e.g., SD 1.5 VAE with SDXL model) | Download the matching VAE or use the VAE embedded in the checkpoint |
| Custom node fails to load | Missing Python dependencies | Open ComfyUI Manager → Install Missing Custom Nodes; check node's README |
| `MPS` out of memory | Image resolution too high for available RAM | Reduce to 768×768; add `--lowvram` flag to the startup command |
| Node graph connections show red lines | Output type doesn't match input type | Red = type mismatch (e.g., connecting MODEL port to VAE input) — reconnect to correct port |
| ComfyUI crashes on startup | Python/PyTorch version conflict | Run `pip install --upgrade torch torchvision` in the ComfyUI venv |
| Slow startup (> 30 sec) | Large number of models in checkpoints/ | Normal for 10+ model files — ComfyUI hashes all of them on startup |

### MPS unified memory vs VRAM

Apple Silicon does not have separate VRAM — the GPU and CPU share the same physical memory pool. When ComfyUI reports "Total VRAM 0 MB", this is expected behavior. The relevant number is your total unified memory (16, 32, 64, or 96 GB). The `--lowvram` flag tells ComfyUI to aggressively offload model layers to system memory during generation, allowing larger models to run on 16 GB systems at the cost of speed:

```bash
python main.py --force-fp16 --lowvram
# Use when: generation fails with memory errors at 16 GB unified memory
# Trade-off: generation speed drops by 2-4× as layers are swapped in/out
```

### Fixing VAE mismatch (black images)

Different model families use different VAEs. If your generated image is solid black, the VAE component is mismatched. Fix by loading the correct VAE explicitly:

```
1. Download the matching VAE to ComfyUI/models/vae/
   For SDXL: huggingface-cli download stabilityai/sdxl-vae sdxl_vae.safetensors --local-dir ./models/vae
2. In the node graph: right-click → Add Node → loaders → VAELoader
3. Connect the VAELoader output to the KSampler's "vae" input (disconnecting the checkpoint's VAE)
4. Select "sdxl_vae.safetensors" in the VAELoader dropdown
```

### Installing custom nodes with ComfyUI-Manager

ComfyUI-Manager is the standard way to install community extensions like ControlNet, AnimateDiff, and upscalers:

```bash
# Install ComfyUI-Manager
cd ComfyUI/custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git

# Restart ComfyUI — a "Manager" button appears in the sidebar
# Use Manager → Install Custom Nodes to browse and install extensions
# Manager → Update All keeps all custom nodes current
```
