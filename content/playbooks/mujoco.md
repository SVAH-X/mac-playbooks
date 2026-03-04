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

MuJoCo (Multi-Joint dynamics with Contact) is a physics simulation engine developed by DeepMind and now used across robotics research at Google, OpenAI, Meta, and hundreds of academic labs. It simulates rigid body dynamics, contact forces, and actuators at high speed with smooth, differentiable physics — the differentiability is key to why it dominates reinforcement learning research.

MuJoCo MJX is a JAX-based version of the same simulation that runs on GPUs. On Apple Silicon, it runs via jax-metal on the Metal GPU, achieving 650,000 simulation steps per second on M3 Max. MuJoCo Playground (RSS 2025 Outstanding Demo Paper Award) provides ready-made RL environments for locomotion and manipulation tasks with proven zero-shot sim-to-real transfer — you train a policy in simulation and deploy it to a physical robot without any adaptation.

Isaac Sim requires NVIDIA GPUs and cannot run on macOS. MuJoCo is the practical alternative for Mac-based robotics research.

## What you'll accomplish

MuJoCo installed and the 3D interactive viewer working. A Python script running the humanoid model at 650K+ steps/sec using MJX on the Metal GPU. A brief reinforcement learning training run with MuJoCo Playground on the Go2 quadruped locomotion task, with training curves showing reward improvement.

## What to know before starting

- **Rigid body dynamics**: MuJoCo numerically integrates Newton's equations for systems of connected rigid bodies. Each timestep solves for contact forces, joint torques, and resulting accelerations.
- **Differentiable contact**: MuJoCo's contact model produces smooth gradients through contact events — essential for gradient-based RL algorithms that backpropagate through simulation.
- **MJX**: The JAX-compiled version of MuJoCo. Running `jax.jit(mjx.step)` compiles the physics step to a Metal GPU shader. `jax.vmap` vectorizes it across thousands of parallel environments in a single Metal dispatch.
- **Sim-to-real transfer**: A policy trained in MuJoCo simulation and deployed directly to a physical robot without any real-world data or fine-tuning. MuJoCo Playground environments are tuned to minimize the sim-to-real gap.
- **PPO (Proximal Policy Optimization)**: The RL algorithm used in Playground. It collects experience from many parallel environments, computes policy gradient updates with a clipped objective function that prevents large policy updates.

## Prerequisites

- macOS 12.0+, Apple Silicon (M1 or later)
- Python 3.9+
- jax-metal installed (see JAX playbook for setup)
- XQuartz for the 3D viewer (optional): `brew install --cask xquartz`

## Time & risk

- **Duration:** 30 minutes
- **Risk level:** Low — pip installs only, no system modification

<!-- tab: Install -->
## Step 1: Install MuJoCo Python bindings

The `mujoco` Python package ships the MuJoCo C library precompiled for Apple Silicon — no compilation step needed. It includes a set of built-in robot models (humanoid, ant, cheetah, etc.) and the interactive viewer.

`MjModel` holds the static simulation parameters (robot geometry, joint limits, actuator specs read from XML). `MjData` holds the current dynamic state (positions `qpos`, velocities `qvel`, control signals `ctrl`). You reset the simulation by copying a clean `MjData`.

```bash
pip install mujoco

# Verify installation and list built-in models
python -c "
import mujoco
print(f'MuJoCo version: {mujoco.__version__}')
# Load the built-in humanoid model
model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><geom size=\".1\"/></body></worldbody></mujoco>')
print(f'Model loaded: {model.nbody} bodies')
"
```

Expected output: `MuJoCo version: 3.x.x` and `Model loaded: 2 bodies`.

## Step 2: Install MJX (JAX GPU backend)

`mujoco-mjx` provides the JAX-jit-compatible version of the simulation. `put_model()` copies the `MjModel` struct to JAX arrays on the Metal GPU. `put_data()` does the same for `MjData`. After that, `mjx.step()` runs entirely on the GPU — no CPU-GPU data transfer per step.

```bash
pip install mujoco-mjx

# Verify MJX import
python -c "
from mujoco import mjx
import jax
print(f'JAX devices: {jax.devices()}')  # Should show metal device
print('MJX ready')
"
```

You should see a Metal device in the JAX devices list. If you only see `CpuDevice`, refer to the JAX + Metal playbook to install jax-metal.

## Step 3: Verify JAX Metal backend

MJX requires JAX with the Metal plugin to use the GPU. If this isn't configured, MJX will silently fall back to CPU with a 100x performance penalty.

```bash
# Install jax-metal if not already installed
pip install jax-metal

python -c "
import jax
import jax.numpy as jnp

# This should run on Metal GPU
x = jnp.ones((1000, 1000))
result = jnp.dot(x, x)
print(f'JAX default device: {result.device()}')
# Expected: METAL:0 or similar
"
```

## Step 4: Install MuJoCo Playground

Playground is a separate package from DeepMind that provides pre-built RL environments, reward functions, curriculum schedules, and visualization tools for specific robot platforms. Installing it also pulls Brax (Google's RL library built on JAX) and other dependencies.

```bash
pip install playground

# Verify available environments
python -c "
import playground
print(dir(playground))  # Lists available environment modules
"
```

## Step 5: Launch the interactive viewer

The MuJoCo viewer renders the simulation in 3D with contact force visualization, joint angle display, and interactive camera control. On macOS, it uses OpenGL via XQuartz.

```bash
# Option 1: Built-in model browser (no Python needed)
python -m mujoco.viewer

# Option 2: Launch with a specific model file
python -c "
import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path('/path/to/humanoid.xml')
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    for _ in range(2000):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)  # Real-time playback
"
```

If the viewer doesn't open, install XQuartz: `brew install --cask xquartz`, then log out and log back in (XQuartz requires a session restart).

<!-- tab: Simulate -->
## Step 1: Basic CPU simulation

MuJoCo ships with several built-in model files accessible via the `mujoco` package path. The humanoid model has 21 degrees of freedom and 16 actuators — a standard benchmark model for locomotion research.

`mj_step` advances the simulation by `model.opt.timestep` seconds (default 2ms). `viewer.sync()` renders the current state. Running both in a loop gives real-time simulation.

```python
import mujoco
import mujoco.viewer
import numpy as np
import time

# Load the built-in humanoid model
model = mujoco.MjModel.from_xml_path(
    mujoco.utils.get_assets_path() + "/humanoid.xml"
)
data = mujoco.MjData(model)

print(f"Humanoid model:")
print(f"  Bodies: {model.nbody}")
print(f"  Joints (DOF): {model.nq}")         # Generalized coordinates
print(f"  Actuators: {model.nu}")
print(f"  Timestep: {model.opt.timestep}s")

# Run with interactive viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    sim_time = 0.0
    while viewer.is_running() and sim_time < 10.0:  # Run for 10 sim-seconds
        step_start = time.time()

        # Apply random control signals (just to make it move)
        data.ctrl[:] = np.random.uniform(-1, 1, model.nu) * 0.1

        mujoco.mj_step(model, data)
        viewer.sync()
        sim_time += model.opt.timestep

        # Throttle to real-time
        time_until_next = model.opt.timestep - (time.time() - step_start)
        if time_until_next > 0:
            time.sleep(time_until_next)
```

## Step 2: MJX batch simulation on Metal GPU

`jax.vmap` vectorizes a function across a batch dimension — instead of running one simulation, it runs `batch_size` simulations simultaneously in parallel. `jax.jit` compiles the entire step to an optimized Metal GPU shader. The first call triggers JIT compilation (~5-10 seconds); subsequent calls run the compiled shader directly.

```python
import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp
import time

# Load model (same XML as before)
model = mujoco.MjModel.from_xml_path(
    mujoco.utils.get_assets_path() + "/humanoid.xml"
)
data = mujoco.MjData(model)

# Upload model and initial state to Metal GPU
mjx_model = mjx.put_model(model)     # Copies MjModel to JAX arrays on Metal
mjx_data = mjx.put_data(model, data)  # Copies MjData to JAX arrays on Metal

batch_size = 1024  # Parallel environments

# Create batch by replicating initial state (all environments start identical)
# In training, you'd initialize these with different random configurations
def make_batch(single_data, n: int):
    """Replicate a single MjData into a batch of n environments."""
    return jax.vmap(lambda _: single_data)(jnp.arange(n))

batched_data = make_batch(mjx_data, batch_size)

# Compile the step function for batch execution on Metal GPU
# vmap: runs mjx.step on each environment in the batch simultaneously
# jit: compiles the vmapped function to a single Metal shader
@jax.jit
def batch_step(data):
    """Advance all environments by one timestep in parallel."""
    return jax.vmap(lambda d: mjx.step(mjx_model, d))(data)

# Warmup: first call triggers JIT compilation
print("Compiling step function (takes ~10 seconds)...")
batched_data = batch_step(batched_data)
batched_data.qpos.block_until_ready()  # Wait for Metal to finish

# Benchmark: measure steps per second
n_steps = 1000
t0 = time.time()
for _ in range(n_steps):
    batched_data = batch_step(batched_data)
batched_data.qpos.block_until_ready()
elapsed = time.time() - t0

steps_per_sec = batch_size * n_steps / elapsed
print(f"Performance: {steps_per_sec:,.0f} steps/sec")
print(f"  ({steps_per_sec / 1e6:.2f}M steps/sec)")
# Expected on M3 Max: ~650,000 steps/sec
```

## Step 3: Batch initialization with randomization

For RL training, you want each environment in the batch to start with a different initial state — different joint positions, orientations, etc. — to ensure diverse training data.

```python
import jax
import jax.numpy as jnp

def randomize_batch(mjx_data, mjx_model, batch_size: int, rng_key):
    """Initialize batch with randomized joint positions for training diversity."""
    keys = jax.random.split(rng_key, batch_size)

    def init_single(key):
        """Initialize one environment with randomized state."""
        data = mjx_data
        # Randomize joint positions within ±10% of range
        noise = jax.random.uniform(key, shape=data.qpos.shape,
                                    minval=-0.1, maxval=0.1)
        data = data.replace(qpos=data.qpos + noise)
        return data

    return jax.vmap(init_single)(keys)

rng = jax.random.PRNGKey(42)
randomized_batch = randomize_batch(mjx_data, mjx_model, batch_size, rng)
print(f"Batch ready: {batch_size} environments with varied initial states")
```

## Step 4: Benchmark CPU vs Metal GPU

Measure the actual speedup on your Mac to understand the value of the MJX approach.

```python
import mujoco
import time

model = mujoco.MjModel.from_xml_path(
    mujoco.utils.get_assets_path() + "/humanoid.xml"
)
data = mujoco.MjData(model)

# CPU benchmark: single environment
n = 10000
t0 = time.time()
for _ in range(n):
    mujoco.mj_step(model, data)
cpu_sps = n / (time.time() - t0)
print(f"CPU: {cpu_sps:,.0f} steps/sec (single env)")

# Metal GPU benchmark (requires mjx_model, batch_step from Step 2)
n_steps = 1000
t0 = time.time()
for _ in range(n_steps):
    batched_data = batch_step(batched_data)
batched_data.qpos.block_until_ready()
gpu_sps = batch_size * n_steps / (time.time() - t0)
print(f"Metal GPU: {gpu_sps:,.0f} steps/sec ({batch_size} envs)")
print(f"Speedup: {gpu_sps / cpu_sps:.0f}x")
```

<!-- tab: Train -->
## Step 1: Run a Playground training job

Playground's `train` script handles environment setup, PPO training loop, checkpointing, and metrics logging. The `Go2Locomotion` task trains a Unitree Go2 quadruped to walk forward — one of the standard locomotion benchmarks with published sim-to-real transfer results.

`num_envs` is the number of parallel simulation environments used during training. More environments = more diverse experiences per gradient update = faster convergence, but more memory. `num_timesteps` is the total number of environment steps collected for training.

```bash
# Train the Go2 quadruped to walk — takes ~10-30 min depending on hardware
python -m playground.train \
  --env_name "Go2Locomotion" \
  --num_envs 4096 \
  --num_timesteps 50_000_000 \
  --log_dir ./runs/go2_locomotion
```

Expected training behavior: episode reward starts near 0, gradually increases over the first 10M steps, and plateaus at a value indicating stable locomotion. On M2 Ultra, this takes ~15 minutes.

## Step 2: Monitor training progress

Playground logs training metrics to TensorBoard. Key metrics to watch: `episode_reward` (should increase), `entropy` (should decrease as policy becomes more deterministic), `value_loss` (should decrease as the value function improves).

```bash
# Install TensorBoard if needed
pip install tensorboard

# Launch TensorBoard (while training is running)
tensorboard --logdir ./runs/go2_locomotion

# Open http://localhost:6006 in your browser
```

You should see reward increasing from near-zero to a stable plateau. If reward stays at 0, the policy is stuck — try reducing the learning rate or increasing `num_envs`.

## Step 3: Visualize a trained policy

After training completes, Playground saves a checkpoint you can load and visualize in the MuJoCo viewer.

```python
import playground
import mujoco
import mujoco.viewer
import time

# Load the trained environment and policy
env = playground.load("Go2Locomotion")
policy = playground.load_policy("./runs/go2_locomotion/checkpoint_final")

model = env.sys.mj_model
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    obs = env.reset()
    while viewer.is_running():
        action = policy(obs)           # Policy inference
        obs, reward, done, info = env.step(action)

        # Sync viewer with current simulation state
        data.qpos[:] = env.pipeline_state.q
        data.qvel[:] = env.pipeline_state.qd
        mujoco.mj_forward(model, data)
        viewer.sync()

        if done:
            obs = env.reset()
        time.sleep(0.01)  # ~100fps visualization
```

## Step 4: Available Playground environments

| Environment | Robot | Task | DOF |
|---|---|---|---|
| Go2Locomotion | Unitree Go2 quadruped | Forward walking | 12 |
| G1Locomotion | Unitree G1 humanoid | Bipedal walking | 23 |
| AlohaTransfer | Aloha dual-arm | Object manipulation | 14 |
| BarkourVB | Google Barkour | Agile locomotion | 12 |

All environments have published zero-shot sim-to-real transfer results — policies trained in MuJoCo are deployed to physical robots without any real-world fine-tuning.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Viewer doesn't open, no error | XQuartz not installed or session not restarted | `brew install --cask xquartz`, then log out and back in |
| `jax-metal not found` error | jax-metal installed but JAX doesn't detect it | `pip install -U jax-metal`; check `jax.devices()` output |
| OOM during batch simulation | `batch_size=1024` exceeds GPU memory | Reduce to `batch_size=256` or `batch_size=512` |
| Metal device not detected by JAX | JAX version incompatibility | Check jax and jax-metal versions match; see JAX playbook |
| Simulation produces NaN values | Timestep too large for model complexity | Reduce `model.opt.timestep` from default 0.002 to 0.001 |
| Playground training crashes immediately | MuJoCo version mismatch with Playground | `pip install mujoco==3.x.x` matching Playground's requirement |
| Viewer shows very low FPS | Too many `viewer.sync()` calls | Call `viewer.sync()` every 10 steps instead of every step |

### XQuartz setup on macOS

XQuartz provides the OpenGL context that MuJoCo's viewer uses on macOS:

```bash
brew install --cask xquartz

# IMPORTANT: log out of your macOS session and log back in
# XQuartz installs a launch agent that requires a session restart

# Verify XQuartz is running
open -a XQuartz
echo $DISPLAY  # Should show something like /private/tmp/com.apple.launchd.xxx/org.macosforge.xquartz:0
```

### Fixing NaN simulation values

NaN values indicate numerical instability — the physics solver diverged. This happens when control signals are too large, the timestep is too long, or contact configurations are degenerate:

```python
import numpy as np

# Check for NaN after each step
for step in range(1000):
    mujoco.mj_step(model, data)
    if np.any(np.isnan(data.qpos)) or np.any(np.isnan(data.qvel)):
        print(f"NaN detected at step {step}")
        print(f"  qpos: {data.qpos}")
        print(f"  ctrl: {data.ctrl}")
        # Reset simulation
        mujoco.mj_resetData(model, data)
        break

# Fix: reduce control magnitude and timestep
data.ctrl[:] = np.clip(data.ctrl, -1, 1)   # Clip controls
model.opt.timestep = 0.001                   # Halve the timestep
```

### Memory management for large batch sizes

When increasing batch_size beyond 1024, monitor Metal GPU memory:

```bash
# In a separate terminal while simulation runs
sudo asitop  # Check memory bandwidth and GPU memory pressure
```

If memory pressure turns yellow or red, reduce batch_size. The humanoid model uses ~2KB per environment state, so 4096 environments is ~8MB of state — well within Metal GPU capacity. The memory pressure comes from activations during JAX JIT compilation.
