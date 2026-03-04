---
slug: nim-equivalent
title: "Ollama API Serving"
time: "5 min"
color: green
desc: "Serve models via OpenAI-compatible API with Ollama"
tags: [inference, api]
spark: "NIM on Spark"
category: inference
featured: false
whatsNew: false
---

<!-- tab: Overview -->
## Basic idea

Ollama's REST API is OpenAI-compatible, meaning any tool built for the OpenAI API — LangChain, Continue.dev, Cursor, Open WebUI, custom Python apps — works with local Ollama models by changing a single base URL. This is the Mac equivalent of NVIDIA NIM: a production-ready model inference microservice with a standard API, running entirely on your hardware.

The API compatibility is deep: Ollama implements the same JSON request/response schemas as OpenAI for `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, and `/v1/models`. The only differences are that the `model` field uses Ollama model names (e.g., `qwen2.5:7b` instead of `gpt-4o`), and the `api_key` field accepts any non-empty string (Ollama validates the header format but ignores the value).

## What you'll accomplish

A locally running OpenAI-compatible API server at `http://localhost:11434/v1` that any OpenAI SDK client can connect to, with multiple models available for switching, embeddings support for RAG pipelines, and optional cross-machine access for running inference on a Mac from a different device on your network.

## What to know before starting

- **The OpenAI API specification** — a JSON REST API where `POST /v1/chat/completions` takes a `messages` array and returns a `choices` array. This standard was adopted by Anthropic, Mistral, Together, and many others — Ollama's compatibility means any code written against this spec works locally.
- **API keys and why Ollama ignores the value** — the OpenAI SDK requires an `api_key` parameter and sends it as an `Authorization: Bearer <key>` header. Ollama checks that the header is present and syntactically valid (non-empty string) but doesn't validate the actual value. Set it to `"ollama"` or any string.
- **Streaming via server-sent events** — when `stream=True`, the server sends tokens as they're generated using the SSE protocol: each token is a `data: {...}` line, and the stream ends with `data: [DONE]`. This enables real-time display of output without waiting for the full response.
- **What embeddings are** — a text embedding is a dense vector (e.g., 768 or 1536 floating-point numbers) that represents the semantic meaning of a text. Similar texts have vectors that are close together in this high-dimensional space. Embeddings power semantic search, RAG (retrieval-augmented generation), and document clustering.
- **Why a separate embedding model** — `nomic-embed-text` is a 137M parameter model fine-tuned specifically to produce high-quality embeddings. Using a chat model for embeddings works but is wasteful: you'd load a 7B model just to get embeddings that a 137M model produces better.

## Prerequisites

- Ollama installed: `brew install ollama` or from [ollama.com](https://ollama.com)
- At least one model pulled (see Setup tab)
- For embeddings: `nomic-embed-text` model pulled
- For cross-machine access: Tailscale installed (optional — see the Tailscale playbook)

## Time & risk

- **Duration:** 5 minutes
- **Risk level:** None — Ollama runs as a user process, not a system service; no ports opened externally by default
- **Rollback:** `pkill ollama` to stop the server

<!-- tab: Setup -->
## Step 1: Start Ollama with network access

By default, `ollama serve` binds to `127.0.0.1:11434` — only accessible from the same machine. Setting `OLLAMA_HOST=0.0.0.0` makes it accessible from other machines on your network (and from Docker containers running locally).

```bash
# Default: accessible only from localhost
ollama serve

# Network-accessible: any machine on your LAN can connect to port 11434
# Security note: only do this on trusted networks; there's no authentication
OLLAMA_HOST=0.0.0.0 ollama serve

# Check that Ollama is running
curl http://localhost:11434                         # should return: "Ollama is running"
```

Ollama is often already running as a background service if you installed the macOS app. Check with `pgrep ollama` — if it returns a PID, it's already running and you don't need to run `ollama serve`.

## Step 2: Pull required models

Pull a chat model for text generation and an embedding model for vector operations. These are separate models — having both available lets you use Ollama as a complete inference backend for RAG applications.

```bash
# Chat model for text generation
ollama pull qwen2.5:7b               # 7B parameter Qwen model, ~4.7 GB

# Embedding model for semantic search and RAG
# nomic-embed-text: 137M params, 768-dim embeddings, optimized for retrieval
ollama pull nomic-embed-text         # ~274 MB — much smaller than a chat model

# Verify both are available
ollama list
# NAME                          ID            SIZE   MODIFIED
# qwen2.5:7b                   31cd61cdd89f   4.7 GB  2 hours ago
# nomic-embed-text:latest      0a109f422b47   274 MB  2 hours ago
```

## Step 3: Verify the API is up

The `/v1/models` endpoint returns a list of available models in OpenAI format. This is the standard health check used by tools like Continue.dev and Open WebUI to confirm connectivity.

```bash
# Check the models endpoint — should return JSON with your model list
curl http://localhost:11434/v1/models | python3 -m json.tool

# Expected response format:
# {
#   "object": "list",
#   "data": [
#     {"id": "qwen2.5:7b", "object": "model", ...},
#     {"id": "nomic-embed-text:latest", "object": "model", ...}
#   ]
# }
```

If this returns an error, Ollama isn't running. Start it with `ollama serve` in a separate terminal.

<!-- tab: Usage -->
## Step 1: OpenAI Python SDK

The most common integration pattern. Change `base_url` to point to Ollama and set any non-empty `api_key`. The rest of your OpenAI code works unchanged.

```python
from openai import OpenAI

# The only changes from OpenAI cloud: base_url and api_key
client = OpenAI(
    base_url="http://localhost:11434/v1",    # Ollama's OpenAI-compatible endpoint
    api_key="ollama",                        # required field, but Ollama ignores the value
)

response = client.chat.completions.create(
    model="qwen2.5:7b",                      # use Ollama model names, not OpenAI names
    messages=[
        {"role": "system", "content": "You are a concise technical assistant."},
        {"role": "user", "content": "Explain unified memory architecture in 3 sentences."}
    ],
    temperature=0.7,                         # works the same as OpenAI
    max_tokens=200,                          # cap generation length
)

print(response.choices[0].message.content)
# Unified memory architecture places CPU and GPU on the same memory fabric...
```

## Step 2: Streaming responses

Streaming delivers tokens as they're generated, enabling real-time display. The async client handles the event stream — each chunk contains one or a few tokens.

```python
import asyncio
from openai import AsyncOpenAI

async def stream_response():
    client = AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )

    # stream=True activates SSE — response arrives token by token
    stream = await client.chat.completions.create(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": "Write a Python quicksort implementation."}],
        stream=True,                         # returns an async generator
    )

    async for chunk in stream:
        # Each chunk has a delta — the incremental new content
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()  # newline after streaming completes

asyncio.run(stream_response())
```

## Step 3: Embeddings

Generate embeddings for semantic search or RAG pipelines. The `nomic-embed-text` model produces 768-dimensional vectors optimized for retrieval tasks.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# Generate an embedding for a single text
response = client.embeddings.create(
    model="nomic-embed-text",               # use the specialized embedding model
    input="Apple Silicon unified memory architecture",
)

vector = response.data[0].embedding         # list of 768 floats
print(f"Embedding dimension: {len(vector)}")  # 768

# Embed multiple texts at once (batch)
texts = [
    "Machine learning on Apple Silicon",
    "Metal GPU acceleration for ML",
    "Unified memory bandwidth limitations",
]
batch_response = client.embeddings.create(model="nomic-embed-text", input=texts)
embeddings = [item.embedding for item in batch_response.data]  # list of 768-dim vectors

# Compute cosine similarity between first and second text
import numpy as np
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarity = cosine_similarity(embeddings[0], embeddings[1])
print(f"Similarity: {similarity:.3f}")   # ~0.85 — semantically related
```

## Step 4: Concurrent requests

By default, Ollama queues requests and processes them one at a time. For parallel workloads, set `OLLAMA_NUM_PARALLEL` before starting the server.

```bash
# Allow 2 simultaneous requests (doubles memory usage — ensure you have enough RAM)
OLLAMA_NUM_PARALLEL=2 ollama serve

# Or set as environment variable permanently in your shell profile:
# echo 'export OLLAMA_NUM_PARALLEL=2' >> ~/.zshrc
```

```python
import asyncio
from openai import AsyncOpenAI

async def parallel_requests():
    client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    # Send 3 requests concurrently — Ollama will process up to OLLAMA_NUM_PARALLEL at once
    tasks = [
        client.chat.completions.create(
            model="qwen2.5:7b",
            messages=[{"role": "user", "content": f"Answer in one sentence: question {i}"}]
        )
        for i in range(3)
    ]
    responses = await asyncio.gather(*tasks)
    for r in responses:
        print(r.choices[0].message.content)

asyncio.run(parallel_requests())
```

## Step 5: Expose over Tailscale

To use Ollama from another device (phone, laptop, another desktop) via your private Tailscale network, start Ollama bound to the Tailscale interface address. See the Tailscale playbook for initial setup.

```bash
# Get your Tailscale IP
tailscale ip -4                              # example: 100.64.1.5

# Start Ollama bound to Tailscale interface — accessible only from your Tailscale devices
OLLAMA_HOST=100.64.1.5 ollama serve

# From another Tailscale device, connect to:
# base_url="http://100.64.1.5:11434/v1"
```

This is more secure than `0.0.0.0` because only authenticated Tailscale devices can reach the endpoint.

<!-- tab: Troubleshooting -->
## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Connection refused` on port 11434 | Ollama not running | Run `ollama serve` in a terminal, or open the Ollama macOS app |
| `401 Unauthorized` | Client not sending `Authorization` header | Set `api_key` to any non-empty string; the SDK sends it automatically |
| `404 model not found` | Model name typo or model not pulled | Run `ollama list` to see available models; `ollama pull <name>` to add |
| Slow responses with concurrent requests | Only 1 request processed at a time | Set `OLLAMA_NUM_PARALLEL=2` before starting Ollama |
| Streaming `data: [DONE]` causes parse error | Client doesn't handle SSE end-of-stream marker | Use the OpenAI SDK's built-in streaming — don't parse SSE manually |
| CORS error from browser app | Browser blocks cross-origin requests | Set `OLLAMA_ORIGINS=*` env var before starting Ollama |
| Embedding dimension mismatch in vector DB | Mixed embedding models used | Always use the same model for indexing and querying; `nomic-embed-text` = 768 dims |

### Diagnosing connection issues

If you're getting connection refused, work through these checks in order:

```bash
# 1. Is Ollama running at all?
pgrep -l ollama                      # should show ollama process

# 2. What port is it bound to?
lsof -i :11434                       # should show ollama listening on 11434

# 3. Can you reach the base URL?
curl http://localhost:11434          # should return "Ollama is running"

# 4. Can you reach the API endpoint?
curl http://localhost:11434/v1/models  # should return JSON model list

# 5. If step 4 fails but step 3 succeeds, try the native Ollama API as fallback
curl http://localhost:11434/api/tags  # Ollama native endpoint — always available
```

### Fixing CORS errors for browser applications

Browser-based apps (like a custom React frontend) enforce CORS and will block requests to `localhost:11434` by default. Allow cross-origin requests:

```bash
# Allow requests from any origin (development only)
OLLAMA_ORIGINS="*" ollama serve

# Allow requests from a specific origin (production)
OLLAMA_ORIGINS="http://localhost:3000" ollama serve
```

### Understanding the model name format

Ollama model names follow the format `name:tag` where tag is typically a parameter count or quantization variant:

```bash
ollama pull qwen2.5:7b          # 7B parameter model, default quantization
ollama pull qwen2.5:32b         # 32B model
ollama pull qwen2.5:7b-instruct-q4_K_M  # explicit quantization variant

# In your API calls, use the exact name from `ollama list`
# The model field is case-sensitive and must match exactly
```
