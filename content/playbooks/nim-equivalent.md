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

Ollama exposes an OpenAI-compatible REST API at `http://localhost:11434`, letting you use any OpenAI SDK client with your local models. This replaces NVIDIA NIM for local model serving.

## Prerequisites

- Ollama installed and running
- At least one model pulled

## Time & risk

- **Duration:** 5 minutes
- **Risk level:** None

<!-- tab: Setup -->
## Start Ollama server

```bash
ollama serve
```

## Pull a model

```bash
ollama pull qwen2.5:7b
```

## List available models

```bash
ollama list
```

<!-- tab: Usage -->
## OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # required but unused
)

response = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=[
        {"role": "user", "content": "Explain unified memory architecture."}
    ]
)
print(response.choices[0].message.content)
```

## curl

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:7b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Embeddings

```bash
curl http://localhost:11434/api/embeddings -d '{
  "model": "nomic-embed-text",
  "prompt": "Apple Silicon unified memory"
}'
```

<!-- tab: Troubleshooting -->
## Connection refused

Ensure `ollama serve` is running. Check with `curl http://localhost:11434`.

## Model not found

Pull the model first: `ollama pull <model-name>`

## Rate limiting / slow responses

Ollama processes one request at a time by default. For concurrent requests, set `OLLAMA_NUM_PARALLEL=2` before starting the server.
