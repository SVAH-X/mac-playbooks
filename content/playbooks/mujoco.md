---
slug: mujoco
title: "MuJoCo + Playground"
time: "30 min"
color: orange
desc: "Cross-platform robotics simulation — replaces Isaac Sim/Lab"
tags: [robotics, simulation]
spark: "Isaac Sim / Isaac Lab"
category: robotics
featured: false
whatsNew: true
---

<!-- tab: Overview -->
## Basic idea

MuJoCo is the cross-platform robotics simulation alternative to Isaac Sim/Lab. It runs natively on macOS with excellent Apple Silicon performance. MuJoCo Playground (RSS 2025 Outstanding Demo Paper) enables training locomotion and manipulation policies with zero-shot sim-to-real transfer.

## Why MuJoCo over Isaac Sim on Mac?

- Isaac Sim requires NVIDIA GPUs (PhysX, RTX rendering) — cannot run on macOS
- MuJoCo runs natively on macOS with **650K steps/sec on M3 Max**
- MuJoCo MJX (JAX backend) enables GPU-accelerated parallel simulation on Apple Silicon
- Proven zero-shot sim-to-real transfer

## Prerequisites

- macOS 12.0+
- Apple Silicon Mac (M1 or later)
- Python 3.9+

## Time & risk

- **Duration:** 30 minutes
- **Risk level:** Low

<!-- tab: Install -->
## Install MuJoCo

```bash
pip install mujoco
```

## Install MJX (GPU-accelerated parallel simulation)

```bash
pip install mujoco-mjx jax jax-metal
```

## Install Playground (RL training environments)

```bash
pip install playground
```

## Launch the interactive viewer

```bash
python -m mujoco.viewer
```

<!-- tab: Simulate -->
## Basic simulation

```python
import mujoco
import mujoco.viewer

# Load a built-in model
model = mujoco.MjModel.from_xml_path(
    mujoco.utils.get_assets_path() + "/humanoid.xml"
)
data = mujoco.MjData(model)

# Run simulation
with mujoco.viewer.launch_passive(model, data) as viewer:
    for _ in range(1000):
        mujoco.mj_step(model, data)
        viewer.sync()
```

## MJX batch simulation on Metal GPU

```python
import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp

model = mujoco.MjModel.from_xml_path("humanoid.xml")
data = mujoco.MjData(model)

# Put model on Metal GPU
mjx_model = mjx.put_model(model)
mjx_data = mjx.put_data(model, data)

# Batch simulate 1024 environments in parallel
batch_size = 1024
batched_data = jax.vmap(lambda _: mjx_data)(jnp.arange(batch_size))

@jax.jit
def step(data):
    return mjx.step(mjx_model, data)

# Run simulation steps
for _ in range(1000):
    batched_data = jax.vmap(step)(batched_data)

print(f"Simulated {batch_size * 1000} steps")
```

<!-- tab: Train -->
## Train a robot with MuJoCo Playground

```bash
# Train a quadruped (Go2) to walk
python -m playground.train \
  --env_name "Go2Locomotion" \
  --num_envs 4096 \
  --num_timesteps 50_000_000
```

## Available environments

| Environment | Robot | Task |
|---|---|---|
| Go2Locomotion | Unitree Go2 | Quadruped walking |
| G1Locomotion | Unitree G1 | Humanoid walking |
| AlohaTransfer | Aloha | Manipulation |
| BarkourVB | Barkour | Agile locomotion |

## Comparison: Isaac Lab vs MuJoCo

| Feature | Isaac Lab (NVIDIA) | MuJoCo + Playground (macOS) |
|---|---|---|
| Platform | Linux + NVIDIA GPU only | macOS, Linux, Windows |
| Parallel envs | 4,096+ on GPU | 1,024+ via MJX on Apple Silicon |
| Sim-to-real | Excellent | Excellent (proven zero-shot transfer) |
| Photorealistic rendering | Yes (RTX) | No |
