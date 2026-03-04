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

Open WebUI is a self-hosted web application that gives you a polished ChatGPT-style interface on top of Ollama (or any OpenAI-compatible API). Bare Ollama gives you a CLI and a raw HTTP API — useful for developers but not for everyday use. Open WebUI adds:

- Persistent conversation history stored in a local SQLite database
- Model switching from a dropdown without restarting anything
- Document upload and RAG (Retrieval-Augmented Generation) for chatting with your files
- System prompt management and model-specific presets
- Multi-user accounts if you want to share a local server with teammates

On NVIDIA/cloud setups you might use hosted frontends or managed services. On a local Mac setup, Open WebUI running against local Ollama means your conversations, documents, and model weights never leave your machine.

## What you'll accomplish

After following this playbook you will have:

- Open WebUI running at `http://localhost:3000`
- Persistent conversation history that survives restarts
- The ability to chat with any model you have pulled in Ollama via a polished web UI
- An admin account with all data stored locally in a Docker volume

## What to know before starting

- **How Docker containers work:** Docker runs applications in isolated environments called containers. Each container has its own filesystem, but can mount "volumes" — directories on your Mac that persist after the container stops. Open WebUI runs in a container so its Python dependencies don't interfere with your system Python.

- **What host-gateway means:** By default, Docker containers on Mac cannot reach `localhost` of the Mac host — `localhost` inside a container refers to the container itself, not your Mac. The `--add-host=host.docker.internal:host-gateway` flag creates a DNS alias that lets the container reach your Mac's localhost, which is where Ollama is listening.

- **What Docker volumes are:** When you pass `-v open-webui:/app/backend/data`, Docker creates a persistent storage volume named `open-webui`. All of Open WebUI's SQLite database, uploaded documents, and user data live here. The volume persists when you stop or remove the container — you don't lose your conversations.

- **What RAG is:** Retrieval-Augmented Generation means the app searches through your uploaded documents, pulls the relevant passages, and includes them in the prompt context before asking the model to respond. The model doesn't "know" your documents — the relevant text is pasted into the prompt at query time.

- **Where data lives:** Everything is local. Conversations are in a SQLite database inside the Docker volume. Models are in `~/.ollama/models/`. Nothing is uploaded to any external service.

## Prerequisites

- Ollama installed and running (`ollama serve` must be active — test with `curl http://localhost:11434/api/tags`)
- Docker Desktop for Mac installed and running (the whale icon in your menu bar), OR Python 3.11+ for the pip install path
- 8 GB+ unified memory
- At least one model pulled in Ollama (e.g., `ollama pull qwen2.5:7b`)

## Time & risk

- **Duration:** 15 minutes (mostly waiting for Docker image download)
- **Risk level:** Low — entirely containerized; one command removes everything
- **Rollback:** `docker rm -f open-webui && docker volume rm open-webui`

<!-- tab: Install -->
## Step 1: Confirm Ollama is running

Open WebUI needs to reach Ollama before it starts. Verify Ollama is up and has at least one model:

```bash
# Check that Ollama's API is responding
curl http://localhost:11434/api/tags
# Expected: {"models":[{"name":"qwen2.5:7b",...}]}
# If you get "connection refused", run: ollama serve
```

If Ollama is not running, start it first: `ollama serve` (in a separate terminal or as a background service with `brew services start ollama`).

## Step 2: Pull and start the Open WebUI container

This is the core install step. The Docker command pulls the Open WebUI image (~1 GB download) and starts it with the right configuration to reach Ollama on your Mac.

```bash
docker run -d \
  -p 3000:8080 \                                            # map Mac port 3000 to container's port 8080
  --add-host=host.docker.internal:host-gateway \            # lets container reach Mac localhost
  -v open-webui:/app/backend/data \                         # persist all app data in a named volume
  --name open-webui \                                       # give the container a memorable name
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \    # tell Open WebUI where Ollama is
  --restart always \                                        # auto-start when Docker Desktop launches
  ghcr.io/open-webui/open-webui:main
```

What each flag does:
- `-d` — run in detached (background) mode; the container runs without blocking your terminal
- `-p 3000:8080` — your Mac's port 3000 maps to the container's internal port 8080
- `--add-host` — creates a hostname `host.docker.internal` inside the container that resolves to your Mac's IP; this is the bridge that lets Open WebUI reach Ollama
- `-v open-webui:/app/backend/data` — mounts a persistent Docker volume so conversations survive container restarts
- `-e OLLAMA_BASE_URL` — environment variable that tells Open WebUI the URL of your Ollama server
- `--restart always` — container automatically restarts if it crashes or if Docker Desktop restarts

## Step 3: Wait for first-time startup

The first launch takes 1–2 minutes because Open WebUI initializes its SQLite database and downloads some frontend assets. Check the logs to see when it's ready:

```bash
# Watch the startup logs in real time
docker logs -f open-webui

# You are looking for this line to confirm it is ready:
# INFO:     Application startup complete.
# Press Ctrl+C to stop watching logs (the container keeps running)
```

If the container crashes immediately, it usually means Ollama is not reachable. Check the logs for error messages mentioning the connection URL.

## Step 4: Create your admin account

Navigate to `http://localhost:3000` in your browser. You will see a signup page.

The first account you create becomes the administrator account with full access to settings, user management, and model configuration. All data stays entirely on your Mac — there is no "Ollama account" or cloud sync involved.

```
1. Go to http://localhost:3000
2. Click "Sign up"
3. Enter a name, email, and password (these are local-only credentials)
4. Click "Create Account"
5. You should land on the chat interface with your Ollama models in the model dropdown
```

## Alternative: pip install (no Docker required)

If you cannot or don't want to run Docker, Open WebUI has a pip package. This installs directly into your Python environment:

```bash
# Requires Python 3.11+
pip install open-webui

# Start the server
open-webui serve --port 3000

# Expected output:
# INFO:     Started server process
# INFO:     Waiting for application startup.
# INFO:     Application startup complete.
# INFO:     Uvicorn running on http://0.0.0.0:3000
```

When to prefer pip over Docker: if you want to use Open WebUI with a virtual environment, if Docker Desktop is too heavy for your workflow, or if you want to install it as a system service using launchd. The pip path is slightly more complex to keep updated (`pip install --upgrade open-webui`).

<!-- tab: Usage -->
## Switching models

The model selector is at the top center of the chat interface. Click it to see all models currently available in Ollama. Switching models mid-conversation starts a new session with that model — the conversation history stays but the new model does not see the prior context unless you explicitly include it.

To add more models to the dropdown, pull them in Ollama: `ollama pull llama3.2:3b`. They appear in Open WebUI automatically.

## System prompts and personas

You can set a persistent system prompt for any model by going to the model name → Edit → System Prompt. This is useful for creating personas (e.g., "You are a senior Go engineer. Be concise.") that persist across all conversations with that model.

## Document upload and RAG

Open WebUI supports RAG — you upload documents and then query them in chat. How it works:

1. Click the paperclip icon in the chat input to attach files (PDF, txt, docx, md)
2. Open WebUI splits the document into chunks and stores embeddings in its local vector database
3. When you ask a question, the app retrieves the most relevant chunks and includes them in the prompt context
4. The model generates an answer grounded in the actual document text

The model does not permanently "learn" your documents. Retrieval happens at query time. This means responses are grounded in your documents without any fine-tuning.

## Using the Open WebUI API

Open WebUI itself exposes an API at `http://localhost:3000/api`. You can use it programmatically with your API key from Settings → Account → API Key:

```bash
curl http://localhost:3000/api/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:7b",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

<!-- tab: Troubleshooting -->
## Quick reference

| Symptom | Likely cause | Fix |
|---|---|---|
| Open WebUI shows "Ollama not connected" | Ollama not running or wrong URL | Confirm `ollama serve` is running; check `OLLAMA_BASE_URL` env var |
| Container exits immediately on `docker run` | Docker Desktop not running or low memory | Open Docker Desktop; set Resources → Memory to 4 GB+ |
| "No models available" in model dropdown | No models pulled in Ollama | Run `ollama pull qwen2.5:7b` on the host |
| Slow first load (30+ seconds) | Container initializing DB on first run | Wait for "Application startup complete" in docker logs |
| Blank screen after login | Browser cache issue or JS error | Hard refresh (Cmd+Shift+R) or try a different browser |
| Port 3000 already in use | Another process on port 3000 | Change port: `-p 3001:8080` in the docker run command |
| Permission denied on volume | Docker Desktop file sharing not enabled | Docker Desktop → Settings → Resources → File Sharing |
| Model list empty after Ollama update | Ollama's model registry refreshed | Restart the Open WebUI container: `docker restart open-webui` |

## Open WebUI cannot connect to Ollama

This is the most common issue and has a specific cause on Mac: Docker containers cannot reach the host's `localhost` without the `--add-host` flag.

Diagnose from inside the container:

```bash
# Shell into the running container
docker exec -it open-webui sh

# Try to reach Ollama from inside the container
curl http://host.docker.internal:11434/api/tags
# If this fails, the host-gateway mapping is missing or Docker Desktop networking is broken

# Exit the container
exit
```

If the above curl fails, recreate the container with the correct `--add-host` flag. First remove the existing container (data is safe — it's in the volume):

```bash
docker rm -f open-webui

# Then re-run the docker run command from Step 2 with --add-host included
```

## Container keeps restarting (restart loop)

If `docker ps` shows the container in a restart loop, check the logs for the actual error:

```bash
docker logs open-webui --tail 50
```

Common causes:
- Port 8080 inside the container conflicts (unusual but check for other containers)
- Volume permissions issue — fix with `docker volume rm open-webui` (WARNING: deletes all conversation history) then re-run

## Updating Open WebUI

The `main` tag always points to the latest release. To update:

```bash
# Pull the latest image
docker pull ghcr.io/open-webui/open-webui:main

# Remove and recreate the container (volume data is preserved)
docker rm -f open-webui

# Re-run the original docker run command
# Your conversations and account are in the volume and will be restored
```