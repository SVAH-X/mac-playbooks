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

LM Studio is a native macOS application that wraps MLX (and llama.cpp for non-MLX models) with a polished interface for browsing, downloading, and running models. It abstracts away all the command-line complexity of MLX LM while still using it as the inference backend — meaning you get the same performance as running `mlx_lm.generate` directly, but through a GUI.

Think of it this way: MLX LM gives you maximum control and scriptability; LM Studio gives you the same inference speed with a point-and-click interface and no terminal required. For daily use, model exploration, or sharing local AI access with less technical teammates, LM Studio is often the better choice over the raw CLIs.

The application has two main components:
1. A model library browser and downloader connected to Hugging Face
2. A local server that exposes an OpenAI-compatible API at `localhost:1234` — the same interface used by tools like Cursor, Continue.dev, and Obsidian Copilot

On Apple Silicon, LM Studio prioritizes the MLX backend for any model that has an MLX variant. For models that only exist as GGUF files, it falls back to its bundled llama.cpp engine with Metal acceleration.

## What you'll accomplish

After following this playbook you will have:

- LM Studio installed as a native Mac app
- At least one model downloaded (MLX variant for best performance)
- A working chat interface with persistent conversation history
- An OpenAI-compatible local server running at `localhost:1234` that external tools can connect to

## What to know before starting

- **MLX vs GGUF in LM Studio's model search:** When you search for a model in LM Studio, results show both MLX variants and GGUF variants. MLX variants run through Apple's MLX framework and are significantly faster on Apple Silicon. GGUF variants run through llama.cpp's Metal backend — still GPU-accelerated but typically 10–30% slower than MLX. Always choose MLX when available.

- **What the Q suffix means for GGUF models:** In LM Studio's model browser, GGUF models show suffixes like Q4_K_M, Q5_K_M, Q8_0. These indicate quantization level. Q4_K_M is the standard recommendation — good quality, fits a 7B model in ~6 GB RAM. Q8_0 is near-lossless but uses ~2x the RAM. If you see an MLX and a GGUF variant, pick MLX.

- **How LM Studio stores models:** Models are downloaded to `~/Documents/LM Studio/Models/` by default. Each MLX model is typically 3–20 GB depending on parameter count and quantization. You can change the storage location in LM Studio's settings.

- **The local server vs chat interface:** LM Studio has two separate concerns: the chat UI (for you to talk to models directly in the app) and the developer server (an HTTP API for other apps to use). You load a model for chat separately from loading a model for the server. Both can run the same model simultaneously.

## Prerequisites

- macOS 13.1+ (macOS 14 Sonoma or later recommended for MLX performance)
- Apple Silicon Mac (M1 or later) — Intel Macs are supported but will not use the MLX backend
- 8 GB+ unified memory (16 GB recommended for 7B models)
- ~2 GB free disk space per model (varies: 7B 4-bit MLX is ~4.3 GB, 7B GGUF Q4_K_M is ~4.7 GB)

## Time & risk

- **Duration:** 5 minutes to install and first chat (not counting model download time)
- **Risk level:** None — standard macOS app, no system modifications
- **Rollback:** Drag LM Studio to Trash; delete `~/Documents/LM Studio/` to remove all models and data

<!-- tab: Install -->
## Step 1: Download LM Studio

Two download options exist: the Mac App Store version and the direct download from lmstudio.ai. The direct download is recommended because it receives MLX support and new features before the App Store version due to Apple's review process.

```
1. Go to https://lmstudio.ai
2. Click "Download for Mac" — verify it shows "Apple Silicon" (not Rosetta/Intel)
3. Open the .dmg file
4. Drag LM Studio to your Applications folder
5. Open LM Studio from Applications
```

On first launch, macOS may show a Gatekeeper warning ("LM Studio can't be opened because it is from an unidentified developer"). If so:

```
System Settings → Privacy & Security → scroll down to "LM Studio was blocked" → click "Open Anyway"
```

Alternatively, right-click the app icon and select "Open" — this bypasses Gatekeeper for apps downloaded outside the App Store.

Verify you are running the Apple Silicon native build (not Rosetta): once the app opens, check Activity Monitor — find LM Studio in the list and look at the "Kind" column. It should say "Apple" not "Intel" or "Rosetta."

## Step 2: Discover and search models

The Search tab (magnifying glass icon in the left sidebar) is LM Studio's model browser. It connects to Hugging Face and displays curated popular models with metadata about size, quantization, and format.

```
1. Click the magnifying glass icon in the left sidebar
2. Type a model name: "Qwen 2.5 7B Instruct"
3. In the search results, look at the format column:
   - MLX variants show a purple "MLX" badge — prefer these on Apple Silicon
   - GGUF variants show quantization codes (Q4_K_M, Q5_K_M, etc.)
4. Filter to only show MLX models using the "MLX" filter button at the top
```

Understanding the model list:
- Models with "Instruct" or "Chat" in the name are fine-tuned for conversation
- Base models (no Instruct suffix) are pretrained models that complete text but don't follow instructions well — use instruct variants for chat
- Larger parameter counts (14B, 32B) are more capable but require more RAM

## Step 3: Download a model

Once you find a model you want, select the specific variant and download it. The choice between MLX and GGUF matters for performance.

```
1. Click a model in search results to expand its variants
2. Select an MLX variant (e.g., "Qwen2.5-7B-Instruct-4bit" from mlx-community)
   - If no MLX variant exists, choose "Q4_K_M" GGUF as the next best option
3. Click the download arrow next to the variant
4. Watch the progress bar — a 7B 4-bit model downloads ~4.3 GB
```

If you want to change the storage location (e.g., to an external SSD with more space):

```
LM Studio → Settings (gear icon) → "Change models folder" → select your preferred location
```

Models are stored in subdirectories organized by Hugging Face org/model. You can inspect them at the path shown in Settings.

## Step 4: Start a chat session

```
1. Click the Chat tab (speech bubble icon) in the left sidebar
2. At the top of the chat area, click the model selector dropdown
3. Choose your downloaded model — LM Studio loads it into memory (takes 2–5 seconds)
4. Type a message and press Enter or click Send
```

Key settings available in the right sidebar during chat:
- **Temperature** — controls randomness (0 = deterministic, 1 = creative, >1 = unpredictable). Default 0.8 works well for general chat.
- **Context Length** — how many tokens of conversation history the model sees. Larger values use more RAM. 4096 is a good default.
- **System Prompt** — text prepended before the conversation. Use this to give the model a persona or task focus.
- **Preset presets** — LM Studio ships with system prompt presets for common use cases (coding assistant, creative writing, etc.)

Conversations are saved automatically and appear in the left sidebar. They persist between app restarts.

<!-- tab: API Usage -->
## Step 1: Enable the local server

The Developer tab (diamond icon) is where you manage the local API server. The server is separate from the chat interface — you can run both simultaneously with the same model.

```
1. Click the diamond icon in the left sidebar (Developer tab)
2. In the "Select a model to load" dropdown, choose your downloaded model
   - This loads the model specifically for the server (can be the same model as chat)
3. Click "Start Server"
4. You should see: "Server running at http://localhost:1234"
```

Important: the model must be explicitly loaded in the Developer tab for the server to work. A model loaded for chat is not automatically available to the server. LM Studio can run the same model for both, but you must explicitly load it in each context.

CORS settings: if you are building a web app that needs to call `localhost:1234` from the browser, enable CORS in the server settings. For desktop apps (Cursor, Continue.dev), CORS is not needed.

## Step 2: Test the server endpoint

Confirm the server is working before connecting other tools:

```bash
# List models currently loaded in the server
curl http://localhost:1234/v1/models
# Expected: {"data":[{"id":"qwen2.5-7b-instruct","..."}]}

# Send a chat completion request
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b-instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is Apple Silicon?"}
    ],
    "temperature": 0.7,
    "max_tokens": 200
  }'
```

The model ID in requests should match what LM Studio shows in the Developer tab. If the ID does not match exactly, you will get a 404 or the request will be ignored.

## Step 3: Use from Python with the OpenAI SDK

LM Studio's server is fully compatible with the OpenAI Python SDK. You only need to change `base_url` and `api_key`:

```python
from openai import OpenAI

# Connect to LM Studio instead of OpenAI
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",   # LM Studio ignores the key; SDK requires a non-empty string
)

# This code is identical to how you'd use OpenAI's API
response = client.chat.completions.create(
    model="qwen2.5-7b-instruct",   # use the model ID shown in LM Studio's Developer tab
    messages=[
        {"role": "system", "content": "You are a Python code reviewer. Be concise and specific."},
        {"role": "user", "content": "Review this function:\ndef add(a, b):\n    return a + b"},
    ],
    temperature=0.3,   # lower temperature for more consistent code reviews
    max_tokens=500,
)
print(response.choices[0].message.content)

# Streaming example
stream = client.chat.completions.create(
    model="qwen2.5-7b-instruct",
    messages=[{"role": "user", "content": "Explain Python decorators in detail."}],
    stream=True,
    max_tokens=1024,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

## Step 4: Connect external apps

LM Studio's `localhost:1234` endpoint works as a drop-in OpenAI replacement for many developer tools.

**Cursor AI:**
```
Cursor Settings → Models → Add OpenAI API key base URL
Base URL: http://localhost:1234/v1
API Key: lm-studio (any non-empty value)
Model: qwen2.5-7b-instruct (match LM Studio's model ID exactly)
```

**Continue.dev (VS Code extension):**
```json
// ~/.continue/config.json
{
  "models": [
    {
      "title": "LM Studio Local",
      "provider": "lmstudio",
      "model": "qwen2.5-7b-instruct",
      "apiBase": "http://localhost:1234/v1"
    }
  ]
}
```

**Obsidian Copilot:**
```
Obsidian Copilot plugin settings:
Provider: OpenAI Compatible
Base URL: http://localhost:1234/v1
API Key: lm-studio
Model: qwen2.5-7b-instruct
```

<!-- tab: Troubleshooting -->
## Quick reference

| Symptom | Likely cause | Fix |
|---|---|---|
| App won't open (Gatekeeper blocked) | Downloaded outside App Store | Right-click → Open, or System Settings → Privacy & Security → Open Anyway |
| Model download stuck at 0% or stalls | Network issue or disk full | Check `~/Documents/LM Studio/Models/` disk space; restart the download |
| Server returns 503 Service Unavailable | No model loaded in Developer tab | Go to Developer tab, select model, click "Start Server" |
| Slow inference (5–10 tok/s on 7B) | Using GGUF model instead of MLX | In model browser, filter to MLX and download the MLX variant |
| High memory usage (multiple models) | Multiple models loaded for chat | In Chat tab, click the model name → "Eject" to unload models not in use |
| "Context length exceeded" error | Input + conversation exceeds model's context window | Start a new conversation, or reduce context length in the right sidebar |
| Blank chat after switching models | Model failed to load (often OOM) | Check LM Studio's logs (View → Toggle Developer Tools → Console); try a smaller model |
| Disk full during download | Model larger than available space | Free space or change models folder to a larger drive in Settings |

## App blocked by Gatekeeper

macOS Gatekeeper blocks apps downloaded from outside the App Store by default. LM Studio is notarized by Apple but may still require manual approval on first launch.

```
Option 1: Right-click the LM Studio icon in Applications → select "Open"
           (bypasses Gatekeeper for this launch and future launches)

Option 2: System Settings → Privacy & Security
           Scroll to the "Security" section
           Look for "LM Studio was blocked from use because it is not from an identified developer"
           Click "Open Anyway"

Option 3 (terminal):
xattr -d com.apple.quarantine /Applications/LM\ Studio.app
```

## Server returns 503 (model not loaded)

A 503 from `localhost:1234` means the server is running but no model is loaded. LM Studio requires you to explicitly load a model in the Developer tab before the server will accept inference requests.

```
1. Open LM Studio → Developer tab (diamond icon)
2. Find the model selector at the top
3. Select a model from the dropdown
4. Wait for the loading indicator to complete (2–10 seconds)
5. The status bar should show "Model loaded" and the model name
6. Re-test with: curl http://localhost:1234/v1/models
```

If the model fails to load, check LM Studio's logs via View → Toggle Developer Tools → Console tab. Common causes: not enough RAM, corrupted model file (re-download it), or the model format is not supported.

## Wrong inference backend (using GGUF instead of MLX)

If your tokens/sec is much lower than expected, you may have accidentally downloaded a GGUF variant instead of an MLX variant. GGUF uses the llama.cpp Metal backend, which is still GPU-accelerated but typically 10–30% slower than MLX.

How to check which backend is being used:

```
LM Studio → Developer tab → look at the loaded model name
- MLX models show [MLX] prefix or are from mlx-community
- GGUF models show quantization codes like Q4_K_M in the name

In the chat area, look at the status bar during generation:
- "MLX" indicates the MLX backend
- "llama.cpp" indicates the GGUF backend
```

To switch to MLX:

```
1. Eject the current model (click model name → Eject)
2. Go to Search tab → filter to "MLX"
3. Download the MLX version of the same model
4. Load it in Developer tab and re-benchmark
```

## Multiple models loaded (high memory usage)

LM Studio keeps loaded models in memory until you explicitly eject them. If you have tried several models, all of them may still be in RAM, causing memory pressure and slow inference.

```
Chat tab → click the model name in the top bar → click "Eject model"
Developer tab → click the loaded model → "Eject"

After ejecting unused models, generation speed on the remaining model should improve.
```

You can also set a memory limit: LM Studio Settings → Advanced → Max RAM for models.