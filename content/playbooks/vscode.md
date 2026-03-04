---
slug: vscode
title: "VS Code"
time: "5 min"
color: green
desc: "Install and configure VS Code for ML development on Mac"
tags: [tools]
spark: "VS Code"
category: tools
featured: false
whatsNew: false
---

<!-- tab: Overview -->
## Basic idea

VS Code is a lightweight, powerful code editor with excellent Python and Jupyter support. The native Apple Silicon build provides full performance on M-series Macs.

## Prerequisites

- macOS 10.15+
- Apple Silicon or Intel Mac

## Time & risk

- **Duration:** 5 minutes
- **Risk level:** None

<!-- tab: Install -->
## Download and install

```bash
# Via Homebrew (recommended)
brew install --cask visual-studio-code
```

Or download the Apple Silicon native build from https://code.visualstudio.com/download

## Verify Apple Silicon build

Help → About → should show "arm64" architecture.

<!-- tab: Extensions -->
## Recommended extensions for ML on Mac

Install from the Extensions panel (Cmd+Shift+X):

- **Python** (ms-python.python) — Python language support
- **Jupyter** (ms-toolsai.jupyter) — Jupyter notebooks in VS Code
- **Pylance** (ms-python.vscode-pylance) — fast Python type checking
- **Continue** (Continue.continue) — local AI coding assistant with Ollama
- **GitLens** (eamodio.gitlens) — enhanced Git integration

## Install via command line

```bash
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
code --install-extension ms-python.vscode-pylance
code --install-extension Continue.continue
```

<!-- tab: Tips -->
## Python interpreter

Select your Python environment: Cmd+Shift+P → "Python: Select Interpreter"

## Jupyter integration

Open any `.ipynb` file directly in VS Code. Notebooks run with the selected Python interpreter.

## Settings for ML development

Add to `settings.json` (Cmd+Shift+P → "Open User Settings JSON"):

```json
{
  "python.defaultInterpreterPath": "~/.venv/bin/python",
  "jupyter.notebookFileRoot": "${workspaceFolder}",
  "editor.formatOnSave": true,
  "python.formatting.provider": "black"
}
```
