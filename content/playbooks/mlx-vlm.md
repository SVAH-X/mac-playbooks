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

MLX VLM brings vision-language model inference to Apple Silicon. Describe images, answer questions about photos, or build multimodal pipelines entirely locally.

## Prerequisites

- macOS 14.0+
- Apple Silicon Mac (M1 or later)
- Python 3.10+
- 16 GB+ unified memory recommended

## Time & risk

- **Duration:** 15 minutes
- **Risk level:** Low — pip install

<!-- tab: Install -->
## Install

```bash
pip install mlx-vlm
```

## Supported models

- Qwen2.5-VL series
- LLaVA series
- Pixtral
- Idefics3
- Many more at [mlx-community](https://huggingface.co/mlx-community)

<!-- tab: Examples -->
## Basic image description

```python
from mlx_vlm import load, generate

model, processor = load("mlx-community/Qwen2.5-VL-7B-Instruct-4bit")

output = generate(
    model, processor,
    "Describe what you see in this image.",
    image="path/to/image.jpg",
    max_tokens=256
)
print(output)
```

## Answer questions about images

```python
output = generate(
    model, processor,
    "What text appears in this image? List all visible text.",
    image="screenshot.png",
    max_tokens=512
)
print(output)
```

## CLI usage

```bash
python -m mlx_vlm.generate \
  --model mlx-community/Qwen2.5-VL-7B-Instruct-4bit \
  --image photo.jpg \
  --prompt "What is shown in this image?"
```

<!-- tab: Troubleshooting -->
## Image not loading

Ensure the image path is absolute or relative to your working directory. Supported formats: JPEG, PNG, WebP.

## Out of memory

Use the 3B or 2B variant:
```bash
model, processor = load("mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
```

## Model download is slow

Models are downloaded from Hugging Face on first load. Ensure a stable internet connection. Models are cached in `~/.cache/huggingface/`.
