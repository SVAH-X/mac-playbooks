---
slug: lm-studio
title: "LM Studio on macOS"
time: "5 min"
color: green
desc: "Polished GUI for local LLM inference using MLX backend"
tags: [inference, ui]
spark: "LM Studio / SGLang"
category: inference
featured: false
whatsNew: true
---

<!-- tab: Overview -->
## Basic idea

LM Studio is a polished native macOS application for running LLMs locally. It uses the MLX backend for maximum Apple Silicon performance, provides a ChatGPT-like UI, and exposes an OpenAI-compatible local server.

## Prerequisites

- macOS 13.1+
- Apple Silicon Mac (M1 or later)
- 8 GB+ unified memory

## Time & risk

- **Duration:** 5 minutes
- **Risk level:** None — standard app install
- **Rollback:** Drag to Trash

<!-- tab: Install -->
## Download and Install

1. Download from [lmstudio.ai](https://lmstudio.ai) (native Apple Silicon build)
2. Open the .dmg and drag LM Studio to Applications
3. Launch LM Studio

## Download a model

1. Click the **Search** tab (magnifying glass icon)
2. Search for a model (e.g., `Qwen 2.5 7B Instruct`)
3. Select the **MLX** variant for best performance
4. Click **Download**

<!-- tab: API Usage -->
## Start the local server

1. Go to the **Developer** tab (diamond icon)
2. Select your downloaded model
3. Click **Start Server** — starts an OpenAI-compatible API at `http://localhost:1234/v1`

## Test the server

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Use with Python

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
response = client.chat.completions.create(
    model="qwen2.5-7b-instruct",
    messages=[{"role": "user", "content": "What is Apple Silicon?"}]
)
print(response.choices[0].message.content)
```

<!-- tab: Troubleshooting -->
## App won't open

Check macOS version (requires 13.1+). Try right-clicking → Open if Gatekeeper blocks it.

## Model download fails

Check disk space. LM Studio stores models in `~/Documents/LM Studio/Models/` by default.

## Server not responding

Ensure the model is loaded and the server is started in the Developer tab.
