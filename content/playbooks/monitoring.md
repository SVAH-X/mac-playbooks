---
slug: monitoring
title: "System Monitoring (asitop)"
time: "5 min"
color: green
desc: "Monitor GPU, CPU, ANE, and memory on Apple Silicon"
tags: [tools, monitoring]
spark: "DGX Dashboard"
category: tools
featured: false
whatsNew: false
---

<!-- tab: Overview -->
## Basic idea

Apple Silicon combines CPU, GPU, Neural Engine (ANE), and memory on a single die in a unified memory architecture. Unlike a desktop with an NVIDIA GPU that has its own dedicated VRAM, your Mac's GPU and CPU share the same RAM pool. This means GPU memory pressure directly competes with system RAM — running a large language model inference can push your Mac to swap even if you have 32GB of RAM.

`asitop` uses Apple's private IOKit framework to read hardware performance counters in real-time — the same data source that Activity Monitor uses, but in a terminal-friendly format. On an NVIDIA machine, you'd run `nvidia-smi`. On Apple Silicon, you run `asitop`.

## What you'll accomplish

`asitop` running in your terminal showing live: GPU utilization %, memory bandwidth (GB/s), ANE power draw, CPU cluster utilization (efficiency + performance cores), and unified memory pressure — everything you need to understand ML workload performance and diagnose bottlenecks.

## What to know before starting

- **Unified memory architecture**: The CPU, GPU, and ANE all access the same physical RAM. When you run a 7B LLM, the model weights live in RAM and the GPU reads them every inference pass. The memory bandwidth (GB/s) is the key bottleneck — not VRAM capacity.
- **Memory bandwidth vs capacity**: A 70B model at 4-bit quantization = ~40GB. You need that much RAM. But the performance bottleneck during inference is bandwidth — how fast the GPU can read the weights from RAM. M3 Max provides 400 GB/s; M3 Pro provides 150 GB/s.
- **Memory pressure**: macOS compresses and eventually swaps RAM to the SSD when physical RAM is full. The pressure gauge (green/yellow/red) reflects this. Yellow = compression active; red = swapping to disk = severe performance degradation.
- **Apple Neural Engine (ANE)**: A dedicated ML accelerator on the chip optimized for CoreML models. PyTorch (MPS) and MLX do NOT use the ANE — they use the GPU. The ANE is used by on-device Siri, autocorrect, and Core ML apps. Seeing high ANE% during LLM inference is unexpected.
- **CPU cluster topology**: M-series chips have two CPU clusters: efficiency (E) cores for background tasks and performance (P) cores for compute. ML frameworks should be using the P cores.

## Prerequisites

- macOS 12.0+, Apple Silicon (M1 or later)
- Python 3.9+
- `sudo` access (required for IOKit hardware monitoring APIs)

## Time & risk

- **Duration:** 5 minutes
- **Risk level:** None — read-only hardware monitoring, no system changes

<!-- tab: Install -->
## Step 1: Install asitop

`asitop` is a Python package that uses `psutil` for process monitoring and Apple's `powermetrics` tool (via subprocess) for hardware metrics. It accesses IOKit performance counters that require root privileges — this is why `sudo` is needed at runtime, not at install time.

```bash
# Install as your regular user
pip install asitop

# Verify the binary is accessible
which asitop   # Should show path to asitop binary
asitop --help  # Should print usage (no sudo needed for --help)
```

If `pip install` installs to a location not on your PATH, try `pip3 install asitop` or `python3 -m pip install asitop`.

## Step 2: Run asitop

The `sudo` requirement exists because IOKit hardware monitoring APIs are restricted to root on macOS. This is analogous to how `nvidia-smi` sometimes requires sudo for power monitoring on Linux.

```bash
sudo asitop
```

You should immediately see a terminal dashboard with CPU, GPU, ANE, and memory panels. If you see `command not found`, the install location isn't on sudo's PATH — see the Troubleshooting tab.

Press `q` to quit.

## Step 3: Cross-check with Activity Monitor

Verify that asitop's GPU% roughly matches what Activity Monitor reports:

```
Activity Monitor → Window menu → GPU History
```

The numbers won't be identical (different sampling rates and methodologies) but should be in the same ballpark within ±10%. If asitop shows 80% GPU and Activity Monitor shows 0%, there's a display issue with asitop (usually fixed by updating it).

```bash
# Update to latest version if readings seem wrong
pip install -U asitop
```

<!-- tab: Usage -->
## Step 1: Reading the asitop display

Each panel in asitop represents a different hardware subsystem:

```
┌─ CPU ──────────────────────────────────────┐
│ E-CPU  ████░░░░░░  35%   1.2 GHz           │  ← Efficiency cores (background tasks)
│ P-CPU  ████████░░  78%   3.4 GHz           │  ← Performance cores (your ML workload)
├─ GPU ──────────────────────────────────────┤
│ GPU    ██████████  94%   1.296 GHz         │  ← Metal GPU utilization
├─ ANE ──────────────────────────────────────┤
│ ANE    ░░░░░░░░░░   0%   0.00 W            │  ← Neural Engine (should be 0 for LLMs)
├─ Memory ───────────────────────────────────┤
│ RAM    18.4 / 32.0 GB                      │  ← Physical RAM used
│ BW     ████████░░  245 GB/s                │  ← Memory bandwidth (key bottleneck)
│ Pressure  ██░░░░░  Green                   │  ← Green=healthy, Yellow=compressed, Red=swap
└────────────────────────────────────────────┘
```

- **E-CPU**: Efficiency cluster. Should be low-to-moderate during ML tasks. High E-CPU means background system activity.
- **P-CPU**: Performance cluster. Should be high during ML training, low during pure GPU inference.
- **GPU**: Metal GPU shader utilization. High during LLM inference and image generation.
- **ANE**: Neural Engine. Should be 0% during PyTorch/MLX workloads. Non-zero during CoreML models.
- **Memory BW**: Bandwidth in GB/s. This is the bottleneck for LLM inference — you want to see it near the theoretical maximum (e.g., 400 GB/s on M3 Max for heavy inference).
- **Memory Pressure**: The health indicator for your RAM situation.

## Step 2: Expected readings during ML workloads

Different workloads produce distinct asitop signatures:

**LLM inference with Ollama/MLX (e.g., running qwen2.5:7b):**
- GPU: 70–95% (model computation happening on Metal GPU)
- Memory BW: 60–80% of theoretical max (reading weights from RAM)
- ANE: 0% (MLX and llama.cpp use Metal GPU, not ANE)
- Memory Pressure: Green (7B model uses ~8GB, comfortable on 16GB systems)

**Image generation with ComfyUI (Stable Diffusion):**
- GPU: 85–100% (diffusion steps are GPU-intensive)
- Memory BW: High (moving activations between layers)
- ANE: 0%
- Memory Pressure: Yellow if model is large (SDXL needs ~12GB)

**PyTorch training (MPS backend):**
- GPU: 40–70% (MPS backend is less efficient than MLX for many operations)
- P-CPU: 30–50% (data loading, preprocessing happening on CPU)
- Memory Pressure: Watch carefully during training — gradients double memory usage

**Data processing with pandas/numpy (CPU only):**
- GPU: 0%
- P-CPU: 80–100% across all performance cores
- Memory Pressure: Depends on dataset size

## Step 3: Diagnosing performance issues

Use asitop readings to identify the actual bottleneck in your workload:

**GPU% is low (under 40%) during LLM inference:**
The Metal backend isn't engaged. Check:
```bash
# For Ollama: verify Metal acceleration is active
OLLAMA_DEBUG=1 ollama run qwen2.5:7b "test" 2>&1 | grep -i "metal\|gpu"

# For MLX: check device
python3 -c "import mlx.core as mx; print(mx.default_device())"
# Should show: Device(gpu, 0)
```

**High memory pressure (yellow/red) during inference:**
You're approaching or hitting the RAM limit. Options:
- Use a more quantized model (Q4 instead of Q8 cuts memory in half)
- Close other applications
- Reduce context length if using a chat interface

**High ANE% during LLM inference:**
An unexpected CoreML model is running. Check Activity Monitor for processes using high CPU or CoreML activity. This shouldn't happen with standard Ollama/MLX workflows.

**Memory bandwidth plateau (not increasing during inference):**
The model is too small to saturate the GPU — or another bottleneck (tokenization, Python overhead) is the real limiter. A well-optimized 7B model on M3 Max should use 200-300 GB/s of bandwidth during inference.

## Step 4: Keyboard shortcuts and options

```bash
# Run with custom refresh rate (default is ~1 second)
sudo asitop --interval 0.5   # Refresh every 500ms for smoother display

# Run without color (useful for logging output to a file)
sudo asitop --no-color

# Keyboard shortcuts while running:
# q     — quit
# h     — help overlay
# r     — force refresh
```

<!-- tab: Alternatives -->
## Step 1: powermetrics (built-in, no install)

`powermetrics` is Apple's built-in command-line performance measurement tool. It provides more detailed power consumption data than asitop — useful for comparing power efficiency between model configurations.

```bash
# Sample GPU and CPU power every 1 second
sudo powermetrics \
  --samplers gpu_power,cpu_power \
  --sample-rate 1000 \
  --show-process-coalition

# Sample everything (very verbose)
sudo powermetrics --samplers all -i 2000

# Specific useful samplers:
# gpu_power    — GPU utilization and power draw (watts)
# cpu_power    — Per-core frequency and power
# thermal      — Thermal pressure and throttling status
# network      — Network I/O statistics
```

`powermetrics` output is text-based and scriptable — pipe it to a file for logging performance over a long training run.

## Step 2: Activity Monitor GPU History

Activity Monitor provides a visual GPU history graph without sudo requirements. It's less detailed than asitop but useful for quick checks and for users who prefer a GUI.

```
Applications → Utilities → Activity Monitor
Window menu → GPU History
```

The GPU History window shows a rolling graph of Metal GPU utilization. The CPU and Memory tabs show per-core CPU usage and the memory pressure gauge respectively.

For ML workloads, check:
- **Memory** tab → Memory Pressure graph: should stay green
- **GPU History**: should spike during inference/training
- **CPU** tab: click the "% CPU" column to sort by CPU usage and find what's burning CPU cycles

## Step 3: htop for CPU-only monitoring

When you suspect a CPU bottleneck (slow data loading, Python preprocessing, CPU-only PyTorch operations), `htop` gives a per-core view without the GPU panels.

```bash
brew install htop
htop
```

In htop: press `F6` to sort by CPU usage, `t` to toggle tree view (shows process hierarchy). Look for Python processes consuming multiple cores — this is expected during multi-threaded data loading.

## Step 4: Quick memory check without sudo

For a fast memory status check without launching asitop:

```bash
# Memory pressure and usage summary
vm_stat | head -10

# Current RAM allocation by process (top 10)
top -l 1 -o mem -n 10 | head -20

# Check if swap is active (high pageouts = you've been swapping)
vm_stat | grep "Pages swapped out"
# If this number is increasing over time, you need more RAM or smaller models
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `sudo: asitop: command not found` | sudo doesn't find the pip-installed binary | Use full path: `sudo $(which asitop)` or `sudo python3 -m asitop` |
| `pip install asitop` fails | pip version or Python version issue | Try `pip3 install asitop` or `python3 -m pip install asitop` |
| All readings show 0% or are frozen | macOS version incompatibility or network filesystem | Update asitop: `pip install -U asitop`; disconnect NFS mounts |
| High ANE% during LLM inference | CoreML model is running (unexpected) | Check Activity Monitor for which process is using ANE |
| GPU% stays at 0% during Ollama | Metal GPU not engaged | `OLLAMA_DEBUG=1 ollama run model test` and check for GPU errors |
| `asitop` module not found when using sudo | Pip installed to user path, sudo uses different PATH | Install system-wide: `sudo pip3 install asitop` (or use full path as above) |

### Fixing the sudo PATH issue

The most common asitop problem on macOS is that `sudo asitop` fails because sudo uses a restricted PATH that doesn't include `~/.local/bin` or similar:

```bash
# Find where asitop was installed
which asitop
# Example output: /Users/you/Library/Python/3.11/bin/asitop

# Run using the full path
sudo /Users/you/Library/Python/3.11/bin/asitop

# Or create an alias (add to ~/.zshrc)
alias asitop="sudo $(which asitop)"
```

### Understanding memory pressure colors

macOS's memory pressure has three states that map to how aggressively the system is reclaiming memory:

- **Green**: RAM usage is comfortable. All applications have the memory they need from physical RAM.
- **Yellow**: Memory compression is active. macOS is compressing inactive memory pages to free space. Performance may be slightly impacted.
- **Red**: Swapping to disk is occurring. This causes severe performance degradation — a 10-50x slowdown for memory-intensive operations. Close applications or use smaller models immediately.

```bash
# Check current memory stats in detail
vm_stat
# Look for "Pages compressed" (yellow) and "Pages swapped out" (red indicators)
```
