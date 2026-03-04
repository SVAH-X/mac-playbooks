---
slug: mlx-vlm
title: "MLX VLM Inference"
time: "15 min"
color: green
desc: "Run vision-language models locally with MLX"
tags: [mlx, multimodal]
spark: "Multi-modal Inference"
category: inference
featured: false
whatsNew: false
---

<!-- tab: Overview -->
## Basic idea

Vision-language models (VLMs) accept both images and text as input, enabling visual question answering, image description, document understanding, and multimodal reasoning. MLX VLM runs these models natively on Apple Silicon using the same MLX framework as text-only LLMs.

Here's how it works under the hood: the image is passed through a vision encoder — a Vision Transformer (ViT) that divides the image into patches and converts them into embedding vectors. Those embeddings are projected into the same dimensional space as the language model's token embeddings, then concatenated with the text prompt tokens. The language model then processes both image and text tokens together in its standard attention layers. The model "sees" the image as a sequence of special tokens, each encoding spatial and semantic information about a region of the image.

## What you'll accomplish

A local VLM (Qwen2.5-VL-7B) running on your Mac that can describe images, answer questions about photos, read text in screenshots, and analyze charts — entirely offline with no API calls. Expected throughput: 10–20 tokens/sec on M2/M3 Silicon with the 4-bit quantized model.

## What to know before starting

- **Vision Transformer (ViT)** — a transformer architecture applied to images. The image is split into fixed-size patches (e.g., 14×14 pixels), each patch is flattened into a vector, and the sequence of patch vectors is processed by transformer attention layers. The output is a sequence of image embeddings.
- **Dynamic resolution in Qwen2.5-VL** — unlike older VLMs that resize everything to 336×336, Qwen2.5-VL accepts variable image sizes and tiles large images into multiple crops. A 1920×1080 screenshot might be processed as 6 tiles, each 336×336, giving the model much more detail.
- **Multimodal fusion** — the projection layer between the vision encoder and language model maps image patch embeddings (e.g., 1024-dimensional) to the language model's token dimension (e.g., 4096-dimensional). This is a linear layer trained to align visual and text representations.
- **Image tokens count toward context** — a 336×336 image typically generates 256–1024 image tokens. With dynamic resolution, a high-res image can use 2000+ tokens. This affects `max_tokens` and context window limits.
- **Prefix vs interleaving models** — Qwen2.5-VL is a prefix model: images are processed before text. Some models support interleaving (image tokens mixed throughout). The mlx-vlm API handles this automatically based on model type.

## Prerequisites

- macOS 14.0+ (Sonoma or later)
- Apple Silicon Mac (M1 or later) — Intel not supported
- Python 3.10, 3.11, or 3.12
- `mlx-lm` already installed (`pip install mlx-lm`)
- 16 GB+ unified memory (7B model: ~6 GB for LLM + ~2 GB for vision encoder = ~8 GB total)
- Hugging Face account (free) for model access

## Time & risk

- **Duration:** 15 minutes setup; first model load downloads ~4 GB
- **Risk level:** Low — pip install only, no system configuration
- **Rollback:** `pip uninstall mlx-vlm`; remove `~/.cache/huggingface/hub/mlx-community__Qwen2.5-VL*`

<!-- tab: Install -->
## Step 1: Install mlx-vlm

MLX VLM is a separate package from mlx-lm that adds vision model support. It depends on Pillow for image preprocessing (resizing, format conversion, normalization) and mlx-lm for the language model backend.

```bash
pip install mlx-vlm              # installs mlx-vlm, Pillow, and all dependencies

# Verify the installation
python -c "import mlx_vlm; print('mlx_vlm version:', mlx_vlm.__version__)"
python -c "from PIL import Image; print('Pillow OK')"
```

The install takes 30–60 seconds. If you see a Pillow compilation error, install it separately first: `pip install Pillow==10.4.0`.

## Step 2: Download and test a VLM

The `load()` function downloads the model on first call and caches it in `~/.cache/huggingface/`. It returns two objects: the model (weights loaded into Metal GPU memory) and the processor (handles both text tokenization and image preprocessing).

```python
from mlx_vlm import load, generate

# Downloads ~4 GB on first run, cached on subsequent runs
# mlx-community models are pre-converted to MLX format — no conversion needed
model, processor = load("mlx-community/Qwen2.5-VL-7B-Instruct-4bit")

# The processor is a combined object that:
# 1. Tokenizes text into input_ids
# 2. Resizes and normalizes images into pixel tensors
# 3. Creates attention masks that cover both image and text tokens
print(type(processor))     # Qwen2_5_VLProcessor
print(type(model))         # Qwen2_5_VLModel
```

The first load takes 30–90 seconds (downloading + loading into memory). Subsequent loads take 5–10 seconds (cached weights).

## Step 3: Basic image description

The `generate()` function takes the model, processor, a text prompt, and an image path. The image path can be a local file or a URL. Specify `max_tokens` conservatively — image tokens already consume significant context.

```python
from mlx_vlm import load, generate
from mlx_vlm.utils import load_config

model, processor = load("mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
config = load_config("mlx-community/Qwen2.5-VL-7B-Instruct-4bit")

# Describe a local image
output = generate(
    model,
    processor,
    "Describe what you see in this image in detail.",
    image="path/to/your/photo.jpg",   # accepts .jpg, .png, .webp, .gif
    max_tokens=512,                    # image tokens + output tokens must fit in context
    verbose=False                      # suppress per-token timing output
)
print(output)

# Load from URL instead of local file
output = generate(
    model,
    processor,
    "What objects are in this image?",
    image="https://example.com/photo.jpg",    # URL fetched automatically
    max_tokens=256
)
```

Expected output: a detailed paragraph describing the image content, typically generated at 10–20 tokens/sec on M2/M3 hardware.

## Step 4: CLI usage for quick testing

The `mlx_vlm.generate` module can be called directly from the command line without writing Python. This is useful for one-off image analysis.

```bash
# Quick description of a local image
python -m mlx_vlm.generate \
  --model mlx-community/Qwen2.5-VL-7B-Instruct-4bit \
  --image /path/to/photo.jpg \
  --prompt "What is shown in this image?" \
  --max-tokens 256

# Analyze a screenshot for text content
python -m mlx_vlm.generate \
  --model mlx-community/Qwen2.5-VL-7B-Instruct-4bit \
  --image ~/Desktop/screenshot.png \
  --prompt "Extract all text visible in this screenshot. Format it as a list." \
  --max-tokens 512
```

<!-- tab: Examples -->
## Screenshot text extraction (OCR-like)

Qwen2.5-VL is particularly strong at reading text in images. Unlike traditional OCR, it understands context — it can distinguish a button label from body text and understands layout.

```python
from mlx_vlm import load, generate
from mlx_vlm.utils import load_config

model, processor = load("mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
config = load_config("mlx-community/Qwen2.5-VL-7B-Instruct-4bit")

# Extract text from a screenshot
output = generate(
    model, processor,
    "Extract all text from this screenshot. Preserve the structure — "
    "use headers for section titles and bullet points for list items.",
    image="~/Desktop/screenshot.png",
    max_tokens=1024    # more tokens for text-heavy images
)
print(output)
```

## Chart and graph analysis

VLMs can interpret data visualizations. Be specific in your prompt — ask for values, trends, and comparisons rather than just "describe the chart."

```python
# Analyze a bar chart or graph
output = generate(
    model, processor,
    "This is a performance chart. Answer: "
    "1. What metric is being measured? "
    "2. What are the approximate values for each bar? "
    "3. Which item performs best and by what percentage?",
    image="chart.png",
    max_tokens=400
)
print(output)
```

## Multi-turn conversation with an image

The image is passed on the first turn. Subsequent turns reference the image without re-encoding it — you must manually carry the conversation history.

```python
from mlx_vlm import load, generate
from mlx_vlm.utils import load_config

model, processor = load("mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
config = load_config("mlx-community/Qwen2.5-VL-7B-Instruct-4bit")

image_path = "diagram.png"

# First turn: introduce the image
first_response = generate(
    model, processor,
    "Describe this diagram briefly.",
    image=image_path,
    max_tokens=200
)
print("Turn 1:", first_response)

# Subsequent turns: no image needed — model has already processed it
# Note: mlx-vlm does not natively maintain conversation state;
# for true multi-turn you need to manage chat history manually
# using the processor's apply_chat_template method.
```

## Batch processing multiple images

Process a list of images in sequence. Load the model once and reuse it across all images.

```python
import os
from mlx_vlm import load, generate
from mlx_vlm.utils import load_config

model, processor = load("mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
config = load_config("mlx-community/Qwen2.5-VL-7B-Instruct-4bit")

image_folder = "/path/to/images/"
results = {}

for filename in os.listdir(image_folder):
    if filename.lower().endswith((".jpg", ".png", ".webp")):
        image_path = os.path.join(image_folder, filename)
        description = generate(
            model, processor,
            "Describe this image in one sentence.",
            image=image_path,
            max_tokens=100,
            verbose=False        # suppress timing output for batch runs
        )
        results[filename] = description
        print(f"{filename}: {description}")
```

## Using image URLs instead of local files

Pass an HTTP/HTTPS URL directly — mlx-vlm downloads and caches the image automatically.

```python
output = generate(
    model, processor,
    "What is the main subject of this image?",
    image="https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
    max_tokens=200
)
print(output)
```

<!-- tab: Troubleshooting -->
## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: PIL` | Pillow not installed | `pip install Pillow` then retry |
| `UnidentifiedImageError` | Image format not supported or file corrupted | Convert to JPEG/PNG first: `convert input.bmp output.jpg` (requires ImageMagick) |
| OOM crash with large images | High-resolution images generate many tokens | Resize before passing: `Image.open(path).resize((800, 600)).save(tmp_path)` |
| `ValueError: not a supported model type` | Model is not a VLM or not converted for mlx-vlm | Only use models from mlx-community tagged as VLM; text-only models won't work |
| Very slow inference (< 5 t/s) | Image preprocessing bottleneck or CPU fallback | Check that no layers are on CPU: add `--verbose` to see device placement |
| Output in wrong language | Model follows the prompt's language | Write your prompt in the language you want the response in |
| `NaN in output` after a few tokens | Quantization instability with certain model/image combinations | Try 8-bit: `mlx-community/Qwen2.5-VL-7B-Instruct-8bit` instead of 4-bit |
| Model not found error | Hugging Face rate limit or network issue | Wait 60 seconds and retry; set `HF_TOKEN` env var for authenticated access |

### Resolving image format issues

mlx-vlm supports JPEG, PNG, and WebP natively. If your image is in another format (TIFF, BMP, HEIC), convert it first:

```python
from PIL import Image

# Convert any format to JPEG before passing to the model
def ensure_supported_format(image_path: str) -> str:
    img = Image.open(image_path)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")          # VLMs expect RGB, not RGBA or palette mode
    output_path = image_path.rsplit(".", 1)[0] + "_converted.jpg"
    img.save(output_path, "JPEG", quality=95)
    return output_path

safe_path = ensure_supported_format("input.heic")
output = generate(model, processor, "Describe this image.", image=safe_path, max_tokens=256)
```

### Reducing memory usage for large images

Qwen2.5-VL's dynamic resolution can consume large amounts of context for high-resolution images. If you hit OOM errors, pre-resize the image:

```python
from PIL import Image

def resize_for_vlm(image_path: str, max_side: int = 1024) -> str:
    """Resize image so the longest side is at most max_side pixels."""
    img = Image.open(image_path)
    ratio = min(max_side / img.width, max_side / img.height, 1.0)
    new_size = (int(img.width * ratio), int(img.height * ratio))
    resized = img.resize(new_size, Image.LANCZOS)
    out_path = "/tmp/vlm_resized.jpg"
    resized.save(out_path, "JPEG", quality=90)
    return out_path

output = generate(
    model, processor,
    "What does this image show?",
    image=resize_for_vlm("huge_screenshot.png", max_side=800),  # cap at 800px
    max_tokens=256
)
```

### Choosing a smaller model if memory is constrained

```python
# 3B model: ~3 GB, works on 8 GB machines
model, processor = load("mlx-community/Qwen2.5-VL-3B-Instruct-4bit")

# 7B model: ~8 GB, best quality-to-size ratio
model, processor = load("mlx-community/Qwen2.5-VL-7B-Instruct-4bit")

# 72B model: ~40 GB, requires 64 GB+ machine
model, processor = load("mlx-community/Qwen2.5-VL-72B-Instruct-4bit")
```
