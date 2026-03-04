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

Monitor real-time GPU, CPU, ANE (Apple Neural Engine), and memory usage on Apple Silicon using asitop — the macOS equivalent of nvidia-smi.

## Prerequisites

- macOS 12.0+
- Apple Silicon Mac
- Python 3.9+
- sudo access

## Time & risk

- **Duration:** 5 minutes
- **Risk level:** None

<!-- tab: Install -->
## Install asitop

```bash
pip install asitop
```

## Run

```bash
sudo asitop
```

(sudo is required for IOKit hardware monitoring access)

<!-- tab: Usage -->
## What asitop shows

- **CPU** — usage per cluster (efficiency + performance cores)
- **GPU** — Metal GPU usage and frequency
- **ANE** — Apple Neural Engine utilization
- **Memory** — usage, bandwidth, and memory pressure
- **Power** — CPU and GPU power consumption (watts)

## Key metrics for ML workloads

During LLM inference, you should see:
- High GPU usage (70–95%)
- High memory bandwidth utilization
- Low ANE usage (LLMs don't use ANE)

During image generation (ComfyUI):
- High GPU usage
- Significant memory bandwidth

## Keyboard shortcuts

- `q` — quit
- `h` — help

<!-- tab: Alternatives -->
## powermetrics (built into macOS)

```bash
sudo powermetrics --samplers gpu_power,cpu_power -i 1000
```

## Activity Monitor

Built-in GUI tool: Applications → Utilities → Activity Monitor

Key views:
- **CPU** tab — per-core usage
- **Memory** tab — memory pressure and usage
- **GPU** tab → Window → GPU History — real-time GPU usage

## htop

```bash
brew install htop
htop
```
