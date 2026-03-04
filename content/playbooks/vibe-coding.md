---
slug: vibe-coding
title: "Vibe Coding with Continue.dev"
time: "15 min"
color: green
desc: "Local AI coding assistant with Ollama + VS Code"
tags: [tools, coding]
spark: "Vibe Coding in VS Code"
category: tools
featured: false
whatsNew: false
---

<!-- tab: Overview -->
## Basic idea

Turn your Mac into a local AI coding assistant using Continue.dev + Ollama. Get code completions, chat-based coding help, and code review — all running locally with no data sent to the cloud.

## Prerequisites

- VS Code installed
- Ollama installed and running
- 16 GB+ unified memory recommended

## Time & risk

- **Duration:** 15 minutes
- **Risk level:** None

<!-- tab: Setup -->
## Install Continue extension

1. Open VS Code
2. Press Cmd+Shift+X to open Extensions
3. Search for "Continue"
4. Click Install

## Pull coding models with Ollama

```bash
ollama serve &

# Main chat model (powerful, for complex tasks)
ollama pull qwen2.5-coder:32b

# Autocomplete model (fast, for inline completions)
ollama pull qwen2.5-coder:7b
```

<!-- tab: Configure -->
## Configure Continue

Create or edit `~/.continue/config.yaml`:

```yaml
models:
  - model: qwen2.5-coder:32b
    title: Qwen Coder 32B
    provider: ollama
    apiBase: http://localhost:11434

tabAutocompleteModel:
  model: qwen2.5-coder:7b
  title: Qwen Coder 7B
  provider: ollama
  apiBase: http://localhost:11434

embeddingsProvider:
  provider: ollama
  model: nomic-embed-text
  apiBase: http://localhost:11434
```

## Reload VS Code

Press Cmd+Shift+P → "Developer: Reload Window"

<!-- tab: Usage -->
## Chat with your codebase

- Press Cmd+L to open the Continue chat panel
- Ask questions about your code: "Explain this function"
- Generate code: "Write a Python function that parses JSONL files"
- Select code and press Cmd+L to include it in context

## Inline completions

Tab autocomplete activates as you type. Accept with Tab, reject with Escape.

## Code actions

Select any code, then:
- Cmd+Shift+L → Add to chat
- Right-click → Ask Continue → "Fix this code"
- Right-click → Ask Continue → "Write tests for this"

## Context providers

Add `@file`, `@folder`, or `@codebase` in the chat to include additional context:
```
@file utils.py How can I improve this?
```
