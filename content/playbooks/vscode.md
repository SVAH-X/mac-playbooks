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

VS Code is Microsoft's open-source code editor — the most widely used editor for Python and ML development. Unlike Xcode or PyCharm, it's lightweight, starts in under a second, and its extension ecosystem covers everything from Jupyter notebooks to remote SSH development.

The native Apple Silicon build runs directly on M-series chips without Rosetta 2 emulation, meaning the editor itself is fast and doesn't burn battery unnecessarily. The extension ecosystem provides Python type checking (Pylance), Jupyter notebook support, AI code completion via Continue.dev, and seamless remote development — you can edit files on your Mac while the code runs on a remote Linux GPU server.

## What you'll accomplish

VS Code installed as a native Apple Silicon app, the `code` shell command available in your Terminal, and core extensions installed (Python, Jupyter, Pylance) and connected to your local Python environment for running notebooks and scripts with autocompletion and error highlighting.

## What to know before starting

- **LSP (Language Server Protocol)**: Pylance runs a background type-checking process that analyzes your Python code and provides autocomplete, go-to-definition, and error highlighting in real time. It needs to know which Python interpreter you're using to find your installed packages.
- **Virtual environments**: VS Code works best when you tell it exactly which Python interpreter to use. A project-level venv at `.venv/` is auto-detected. Without a venv, Pylance may report false errors for installed packages.
- **Jupyter kernels**: The Python process that executes notebook cells is called a kernel. VS Code's Jupyter extension can use any installed Python interpreter as a kernel — just select it from the kernel picker in the top-right of a notebook.
- **Command palette**: `Cmd+Shift+P` opens the command palette — a searchable list of all VS Code commands. Nearly every action, including ones with no keyboard shortcut, is accessible here. You'll use it constantly.
- **Workspace settings**: VS Code stores per-project configuration in `.vscode/settings.json`. These override your user-level settings for that project, so you can have different Python paths for different projects.

## Prerequisites

- macOS 10.15+, Apple Silicon or Intel
- Admin rights for installation (standard user install is also possible)

## Time & risk

- **Duration:** 5 minutes
- **Risk level:** None — drag to Trash to uninstall completely

<!-- tab: Install -->
## Step 1: Download the Apple Silicon build

The direct download from code.visualstudio.com is slightly ahead of the Homebrew cask and has no App Store sandbox restrictions. Download the "Apple Silicon" build — not the Universal binary if you're on an M-series Mac (Universal is larger and slightly slower to start).

```bash
# Option A: Homebrew (easiest, manages updates)
brew install --cask visual-studio-code

# Option B: Direct download (visit in browser)
# https://code.visualstudio.com/download
# Click the "Apple Silicon" button under macOS
```

After installation, verify you got the native arm64 build:

```
Help menu → About → look for "Architecture: arm64"
```

If it shows `x86_64`, you downloaded the Intel build — re-download and select "Apple Silicon" explicitly.

## Step 2: Move to Applications and handle Gatekeeper

macOS Gatekeeper will show a security warning the first time you open an app downloaded from the internet. VS Code is signed and notarized by Microsoft, so this is safe to proceed.

1. Drag VS Code from your Downloads folder to `/Applications`
2. Double-click to open — if macOS blocks it, go to System Settings → Privacy & Security → "Open Anyway"

VS Code should open to the Welcome tab. You're done with the install itself.

## Step 3: Install the `code` command in PATH

The `code` CLI lets you open files and folders from Terminal. Without this, you have to drag files to the VS Code dock icon. With it: `code my_project/` opens the entire folder as a workspace.

```
Cmd+Shift+P → type "shell command" → select "Shell Command: Install 'code' command in PATH"
```

Verify it worked:

```bash
# Open a new Terminal window (to reload PATH), then:
code --version
# Expected: 1.xx.x  (version number)

# Test: open the current directory in VS Code
code .
```

## Step 4: Open a Python project

When VS Code opens a folder, it looks for `.venv/`, `venv/`, `.conda/`, and other common venv locations and auto-detects the Python interpreter. Workspace-level settings in `.vscode/settings.json` override your global settings for that project.

```bash
# Create a test project
mkdir ~/test-vscode-project
cd ~/test-vscode-project

# Create a virtual environment (VS Code will auto-detect this)
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas

# Open in VS Code
code .
```

In VS Code: look at the bottom-left status bar — it should show the Python version from your `.venv`. If it shows the system Python, click it to change to your venv interpreter.

<!-- tab: Extensions -->
## Step 1: Install the Python extension

The Python extension (`ms-python.python`) is the foundation. It installs Pylance automatically as a dependency and enables IntelliSense (autocomplete), linting, code formatting, and the Python debugger with breakpoints.

```bash
# Via command line (fastest)
code --install-extension ms-python.python

# Or: Cmd+Shift+X → search "Python" → Install the one by Microsoft
```

After installation, open a `.py` file. You should see:
- Autocomplete as you type
- Red underlines for errors
- The Python interpreter shown in the bottom-left status bar

## Step 2: Install the Jupyter extension

The Jupyter extension (`ms-toolsai.jupyter`) enables opening `.ipynb` files directly in VS Code as interactive notebooks. It also adds the "Interactive Window" — run Python cells in any `.py` file by adding `# %%` cell markers, then pressing `Shift+Enter`.

```bash
code --install-extension ms-toolsai.jupyter

# Install ipykernel in your venv so VS Code can use it as a kernel
source .venv/bin/activate
pip install ipykernel
python -m ipykernel install --user --name myenv
```

Test: create a file called `test.ipynb` in VS Code. A Jupyter notebook opens. Press the kernel selector (top-right) and choose your `.venv` interpreter. Run a cell with `import numpy; print(numpy.__version__)`.

## Step 3: Install Continue.dev for AI assistance

Continue.dev connects VS Code to your local Ollama models, providing tab autocomplete and a chat panel. It's a fully local GitHub Copilot alternative. See the Vibe Coding playbook for complete setup.

```bash
code --install-extension Continue.continue
```

After installation, a Continue icon appears in the left sidebar. You'll configure it to connect to Ollama in the Vibe Coding playbook.

## Step 4: Useful additional extensions

These are all optional but widely useful for ML development:

```bash
# GitLens: enhanced git blame, history, and diff in editor margins
code --install-extension eamodio.gitlens

# Rainbow CSV: color-codes CSV columns for readability
code --install-extension mechatroner.rainbow-csv

# Even Better TOML: syntax highlighting for pyproject.toml
code --install-extension tamasfe.even-better-toml

# indent-rainbow: color-codes indentation levels (helpful for Python)
code --install-extension oderwat.indent-rainbow
```

<!-- tab: Tips -->
## Python interpreter selection

The most common VS Code issue is using the wrong Python interpreter (system Python instead of your venv). Fix this whenever IntelliSense shows errors for packages you've installed:

```
Cmd+Shift+P → "Python: Select Interpreter" → choose ".venv/bin/python (Recommended)"
```

For a project that always uses the same venv, lock it in `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "jupyter.notebookFileRoot": "${workspaceFolder}",
  "editor.formatOnSave": true,
  "editor.rulers": [88]
}
```

## Jupyter keyboard shortcuts

The most important shortcuts for notebook-style development:

- `Shift+Enter` — run cell and move to the next cell
- `Ctrl+Enter` — run cell and stay in place
- `Escape` then `A` — insert cell above
- `Escape` then `B` — insert cell below
- `Escape` then `D D` — delete current cell
- `Cmd+Shift+P → "Jupyter: Restart Kernel"` — restart if kernel hangs

## Integrated terminal

Open the integrated terminal with `Cmd+`` ` (backtick). It opens in the workspace root directory and respects your shell profile. The terminal auto-activates conda or venv environments if you've configured them.

```bash
# The integrated terminal is just a real terminal — run anything
python train.py --epochs 10
jupyter nbconvert --to script notebook.ipynb
```

## Remote development over SSH

VS Code's Remote - SSH extension lets you edit files on a remote Linux machine as if they were local. The extension server runs on the remote machine, so autocomplete and linting work with the remote Python environment.

```bash
code --install-extension ms-vscode-remote.remote-ssh
```

After installing: `Cmd+Shift+P → "Remote-SSH: Connect to Host"` → enter `user@remote-host`. VS Code reconnects automatically on subsequent opens. This is the standard workflow for using VS Code locally while training on a remote GPU server.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| IntelliSense not working (no autocomplete) | Wrong Python interpreter selected | `Cmd+Shift+P → "Python: Select Interpreter"` → choose your venv |
| Jupyter kernel fails to start | `ipykernel` not installed in active venv | `pip install ipykernel` in your venv |
| Extensions not installing | Network issue or proxy | Check internet; try `code --install-extension` from Terminal instead of GUI |
| `code` command not found after install | PATH not updated | Open new Terminal window; or add `/usr/local/bin` to PATH manually |
| Pylance showing wrong errors | Incorrect Python path or stale cache | `Cmd+Shift+P → "Pylance: Clear cache and reload"` |
| Git operations not working | Xcode command line tools missing | `xcode-select --install` |

### Fixing Pylance false errors

If Pylance highlights installed packages as unresolved imports, the interpreter selection is wrong. A definitive fix:

```bash
# In Terminal: confirm the right python path
which python   # Should show .venv/bin/python, not /usr/bin/python3

# In VS Code:
# 1. Cmd+Shift+P → "Python: Select Interpreter"
# 2. Choose the interpreter at "./.venv/bin/python"
# 3. Cmd+Shift+P → "Developer: Reload Window"
```

### Setting up a new ML project from scratch

Recommended project structure that VS Code handles well:

```bash
mkdir my-ml-project && cd my-ml-project

# Create venv (VS Code auto-detects .venv name)
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib jupyter ipykernel

# Create VS Code workspace settings
mkdir .vscode
cat > .vscode/settings.json << 'EOF'
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "jupyter.notebookFileRoot": "${workspaceFolder}"
}
EOF

# Open in VS Code
code .
```

VS Code will detect the `.venv` and configure the Python interpreter automatically.
