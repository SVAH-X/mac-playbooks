---
slug: open-webui
title: "Open WebUI with Ollama"
time: "15 min"
color: green
desc: "Deploy a full ChatGPT-like interface locally"
tags: [inference, ui]
spark: "Open WebUI with Ollama"
category: onboarding
featured: true
whatsNew: false
---

<!-- tab: Overview -->
## Basic idea

Open WebUI gives you a polished ChatGPT-like interface running entirely locally on your Mac, connected to Ollama for model serving.

## Prerequisites

- Ollama installed and running (`ollama serve`)
- Docker Desktop for Mac (recommended) or Python 3.11+
- 8 GB+ unified memory

## Time & risk

- **Duration:** 15 minutes
- **Risk level:** Low — containerized install
- **Rollback:** `docker rm -f open-webui`

<!-- tab: Install -->
## Option 1: Docker (Recommended)

```bash
# Ensure Ollama is running
ollama serve &

# Pull and start Open WebUI
docker run -d -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -v open-webui:/app/backend/data \
  --name open-webui \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```

## Option 2: pip install

```bash
pip install open-webui
open-webui serve --port 3000
```

Open http://localhost:3000 to access the UI.

<!-- tab: Usage -->
## Select a model

1. Open http://localhost:3000
2. Create an account (local only)
3. Select a model from the dropdown at the top
4. Start chatting

## Tips

- Use the **+** button to start new conversations
- Enable **RAG** in settings to chat with documents
- Set a **System Prompt** in the model settings for custom personas

<!-- tab: Troubleshooting -->
## Can't connect to Ollama

Ensure Ollama is running:
```bash
ollama serve
```

Check the OLLAMA_BASE_URL env var points to the correct host.

## Docker container won't start

Check Docker Desktop is running and has sufficient memory allocated (Settings → Resources → Memory: set to 4GB+).
