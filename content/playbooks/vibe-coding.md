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

"Vibe coding" means using an AI assistant to write code by describing what you want in natural language, letting it handle the boilerplate and syntax while you focus on the logic and architecture. Continue.dev is an open-source VS Code extension that connects to your local Ollama models — providing tab autocomplete, an in-editor chat panel, and context-aware code generation, all running locally on your Mac with no data sent to any server.

This is a fully local alternative to GitHub Copilot. The tradeoff is quality vs privacy: a local 32B model is good but not at the level of GPT-4o, while the 7B autocomplete model is fast enough to feel real-time.

## What you'll accomplish

VS Code with Continue.dev configured with two Ollama models: `qwen2.5-coder:32b` for high-quality chat and code generation, and `qwen2.5-coder:7b` for fast inline tab autocomplete. Plus `nomic-embed-text` for the `@codebase` feature that lets Continue search your entire codebase semantically. All responses are local — no API keys, no usage limits, no data leaving your machine.

## What to know before starting

- **LLM inference latency for code completion**: Tab autocomplete needs to feel instant — under 200ms to not break flow. A 7B model responds in ~100-150ms on M2 Pro. A 32B model takes 500-800ms — too slow for autocomplete but fine for chat.
- **Context window**: Continue sends the current file, cursor position, and recently-viewed files as context with each request. Larger context = better suggestions, but more tokens = slower response. Continue automatically trims context to fit the model's window.
- **RAG for codebases**: `@codebase` uses Retrieval-Augmented Generation — Continue indexes your codebase into a local vector database (using the embedding model), then retrieves the most relevant files for each query. This is how it can answer "how is authentication implemented?" over a large codebase.
- **Chat templates**: Coding models are fine-tuned with specific prompt formatting (ChatML, Alpaca, etc.). Ollama handles this automatically — you don't need to configure it, but it's why using the Ollama provider matters.

## Prerequisites

- VS Code installed (see VS Code playbook)
- Ollama running (`ollama serve`)
- `qwen2.5-coder:7b` pulled (for autocomplete)
- 16GB+ unified memory (32B chat model needs ~20GB)

## Time & risk

- **Duration:** 15 minutes
- **Risk level:** None — extension can be disabled or uninstalled from VS Code Extensions panel at any time

<!-- tab: Setup -->
## Step 1: Install the Continue extension

Continue.dev is available in the VS Code marketplace. The extension ID is `Continue.continue`. Installation adds a Continue icon to the left activity bar in VS Code.

```bash
# Install from command line
code --install-extension Continue.continue

# Or in VS Code: Cmd+Shift+X → search "Continue" → Install (publisher: Continue)
```

After installation, click the Continue icon in the left sidebar (looks like a `>>` symbol). The first time you open it, Continue may prompt you to configure a model — skip this and configure manually in Step 2.

## Step 2: Pull the two-model setup

The two-model strategy separates concerns: the 32B model for quality in chat (you wait a few seconds, it's acceptable), and the 7B model for autocomplete speed (must be instant).

`qwen2.5-coder` is preferable to general-purpose `qwen2.5` because it was trained specifically on code data — it knows more about API signatures, design patterns, and common idioms across 40+ programming languages.

```bash
# Chat model: high quality, use for complex questions and generation
# Download size: ~20GB | Memory required: ~22GB
ollama pull qwen2.5-coder:32b

# Autocomplete model: fast responses, use for inline tab completion
# Download size: ~4.7GB | Memory required: ~5GB
ollama pull qwen2.5-coder:7b

# Confirm both are ready
ollama list
```

If you have less than 32GB RAM, use `qwen2.5-coder:14b` for chat (~9GB, good quality) and keep `qwen2.5-coder:7b` for autocomplete.

## Step 3: Pull the embedding model

Continue's `@codebase` feature requires an embedding model to build a local vector index of your code. `nomic-embed-text` is a small (274MB), fast embedding model — it runs almost instantly and produces high-quality code embeddings.

```bash
ollama pull nomic-embed-text

# Verify it's available
ollama run nomic-embed-text "test" 2>/dev/null || echo "Embedding model ready"
```

## Step 4: Verify Ollama is reachable

Before configuring Continue, confirm that Ollama is running and accepting API calls. This is the most common setup issue — if Ollama isn't running, Continue shows "connection failed" with no further detail.

```bash
# Start Ollama if not running
ollama serve &

# Test the API endpoint Continue will use
curl -s http://localhost:11434/api/tags | python3 -m json.tool | grep '"name"'
# Expected: a list of model names including qwen2.5-coder:7b and qwen2.5-coder:32b
```

If `curl` times out, Ollama isn't running. If it returns an empty model list, the models weren't pulled successfully.

<!-- tab: Configure -->
## Step 1: Locate and open config.yaml

Continue uses a YAML configuration file. The user-level config at `~/.continue/config.yaml` applies to all projects. A project-level config at `.continue/config.yaml` in your workspace overrides it for that project (useful when different projects need different models).

```bash
# Create the config directory if it doesn't exist
mkdir -p ~/.continue

# Open in VS Code
code ~/.continue/config.yaml
```

If the file already exists from a previous Continue install, it may be in the older JSON format (`config.json`). The YAML format is current — you can delete `config.json` and create `config.yaml` fresh.

## Step 2: Configure the chat model

Each entry in the `models` list appears in Continue's model selector dropdown. The `provider: ollama` field tells Continue to use the Ollama API at `apiBase`. The `model` field must match exactly what `ollama list` shows.

```yaml
models:
  - model: qwen2.5-coder:32b
    title: Qwen Coder 32B        # Display name in the UI dropdown
    provider: ollama
    apiBase: http://localhost:11434

  # Optional: add a faster option for quick questions
  - model: qwen2.5-coder:7b
    title: Qwen Coder 7B (fast)
    provider: ollama
    apiBase: http://localhost:11434
```

The `title` is what you see in the model selector — name it something meaningful. `apiBase` is where Ollama listens; don't change this unless you've configured Ollama to use a different port.

## Step 3: Configure the autocomplete model

The `tabAutocompleteModel` key is separate from `models` — it's always the model used for inline completions, regardless of which chat model you've selected. Keep this as the 7B model for speed.

```yaml
tabAutocompleteModel:
  model: qwen2.5-coder:7b
  title: Qwen Coder 7B
  provider: ollama
  apiBase: http://localhost:11434
```

Add this block at the same indentation level as `models:` (not inside it).

## Step 4: Configure embeddings for @codebase

The `embeddingsProvider` key tells Continue which model to use when building the codebase vector index. When you run `@codebase` in chat, Continue embeds your query and retrieves the most similar code chunks from this index.

```yaml
embeddingsProvider:
  provider: ollama
  model: nomic-embed-text
  apiBase: http://localhost:11434
```

The full `config.yaml` combining all three sections:

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

## Step 5: Reload and verify

After saving config.yaml, reload VS Code to apply the configuration. Verify the model connected successfully.

```
Cmd+Shift+P → "Developer: Reload Window"
```

Then open the Continue sidebar. You should see:
- A green status indicator next to the model name
- The model dropdown showing "Qwen Coder 32B"
- No error message at the top of the chat panel

To trigger the first autocomplete, open a Python file and start typing a function definition. After a brief pause, a gray suggestion should appear — press Tab to accept.

<!-- tab: Usage -->
## Tab autocomplete

Autocomplete activates automatically as you type. A gray suggestion appears after a short pause (~150ms). There's no special keyboard shortcut to trigger it — just type and wait.

- `Tab` — accept the full suggestion
- `Escape` — dismiss without accepting
- `Ctrl+Right` — accept one word at a time (if the suggestion is long)

Autocomplete works best for:
- Completing function signatures after you write the docstring
- Repeating patterns (e.g., the second item in a list after you wrote the first)
- Boilerplate code (class methods, test functions, argparse arguments)

Autocomplete is less reliable for:
- First-time logic (no pattern to complete)
- Code that requires understanding the broader codebase

## Chat panel (Cmd+L)

Open with `Cmd+L`. The chat panel is a full conversation interface — you can ask multi-turn questions, share code, and have it generate complete files.

```
# Examples of effective chat prompts:

"Write a Python function that reads a JSONL file and yields parsed dicts"

"Explain what this code does and suggest improvements" (select code first, then Cmd+L)

"This function raises a KeyError on line 42 with input {'a': 1}. Why?"

"Add type annotations to this function" (select function, then Cmd+L)

"Write pytest tests for the function I just selected"
```

Context providers in chat — prefix with `@` to include specific context:
- `@file utils.py` — include a specific file
- `@folder src/` — include all files in a folder
- `@codebase how is authentication handled?` — semantic search over your entire codebase

## Code selection actions

Select any code in the editor, then:
- `Cmd+Shift+L` — add selected code to the chat context
- Right-click → "Ask Continue" → choose an action (Fix, Explain, Optimize, Write tests)

This is the primary workflow for refactoring: select a function, `Cmd+Shift+L`, type "refactor to use async/await", review the suggestion, apply.

## Tips for better results

- **Be specific about language and framework**: "Write a FastAPI endpoint that..." is better than "Write a function that..."
- **Include error messages verbatim**: Paste the full stack trace — the model can pinpoint the issue
- **Let autocomplete finish before typing**: If you type while it's thinking, the suggestion is cancelled
- **Use `@codebase` for unfamiliar code**: "How does the database connection pool work in this codebase?" often works better than grep

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| No autocomplete suggestions appear | Ollama not running or model not loaded | Check `ollama list`; run `ollama serve` if needed |
| Chat shows "Ollama connection failed" | Wrong `apiBase` or Ollama not running | Verify `curl http://localhost:11434/api/tags` returns JSON |
| Autocomplete is very slow (>1 second) | Using 32B model for autocomplete | Set `tabAutocompleteModel` to the 7B model |
| `@codebase` returns unrelated files | Index is stale or corrupt | Continue sidebar → Actions → "Rebuild codebase index" |
| Suggestions are in the wrong language | Model doesn't know the file type | Specify language in prompt: "in Python, write..." |
| config.yaml changes not taking effect | VS Code not reloaded after config change | `Cmd+Shift+P → "Developer: Reload Window"` |
| Continue extension crashes VS Code | Extension version incompatibility | Check VS Code version; update Continue extension |

### Rebuilding the codebase index

The `@codebase` feature builds a local vector index in `~/.continue/index/`. If results seem stale or wrong:

```bash
# Option 1: Via Continue UI
# Continue sidebar → gear icon → "Rebuild codebase index"

# Option 2: Delete and rebuild manually
rm -rf ~/.continue/index/
# Reopen VS Code — index rebuilds automatically when you use @codebase
```

Large codebases (>10,000 files) take a few minutes to index. The indexing runs in the background.

### Configuring autocomplete behavior

You can tune how aggressively autocomplete triggers in config.yaml:

```yaml
tabAutocompleteOptions:
  debounceDelay: 150          # Milliseconds to wait after typing stops (default: 150)
  maxPromptTokens: 1024       # Context tokens sent with each completion request
  multilineCompletions: "auto" # "always", "never", or "auto"
```

Lower `debounceDelay` makes it feel more responsive but uses more inference compute. Higher `maxPromptTokens` improves quality but increases latency.
