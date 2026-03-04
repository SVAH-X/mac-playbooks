---
slug: jax-apple
title: "JAX on Apple Silicon"
time: "15 min"
color: green
desc: "Run JAX with Metal GPU backend for scientific computing"
tags: [jax, data science]
spark: "Optimized JAX"
category: data-science
featured: false
whatsNew: false
---

<!-- tab: Overview -->
## Basic idea

JAX is Google's numerical computing library built around composable function transformations. Its four core transforms are: `jit` (JIT compilation to native code), `grad` (automatic differentiation), `vmap` (vectorization across a batch dimension), and `pmap` (parallelization across devices). These transforms compose — you can JIT-compile a vectorized gradient function with `jax.jit(jax.vmap(jax.grad(f)))`.

JAX targets XLA (Accelerated Linear Algebra), a compiler that generates optimized native code for CPUs, GPUs, and TPUs. The `jax-metal` plugin adds an XLA backend for Apple's Metal GPU, enabling JAX programs to run on Apple Silicon without any code changes. This makes JAX the right choice for research-oriented ML on Mac, particularly for MuJoCo MJX (physics simulation that requires JAX) and custom neural network implementations that need composable transforms.

## What you'll accomplish

JAX running on your Mac's Metal GPU with JIT-compiled functions that are measurably faster than their un-compiled equivalents, working automatic differentiation suitable for gradient-based optimization, and a benchmark comparing Metal GPU vs CPU performance.

## What to know before starting

- **JIT compilation**: The first time you call a `@jax.jit`-decorated function, JAX traces your Python code with abstract values to build a computation graph, then compiles it to Metal compute shaders via XLA. This first call takes 1-30 seconds. Subsequent calls with the same input shapes use the cached compiled version and are much faster.
- **Pure functions**: JAX functions must be side-effect-free and must not mutate their inputs. In-place operations (`x[0] = 1.0`) are not allowed inside JIT-compiled code. This is a fundamental constraint, not a limitation that can be worked around.
- **Tracing vs execution**: During JIT tracing, JAX replaces your concrete Python values with abstract "tracers." Code that inspects concrete values (`if x > 0`) will behave differently inside `jit` than outside it — the condition is evaluated at trace time with an abstract value, not at runtime.
- **XLA and Metal**: XLA is a compiler that understands high-level linear algebra operations (matrix multiply, convolution, reduction) and generates optimized code for the target hardware. `jax-metal` teaches XLA how to generate Metal compute shader code for Apple Silicon.
- **Explicit random keys**: JAX has no global random state. Every random operation requires an explicit key (`jax.random.PRNGKey(0)`). To generate different random values, you must split the key: `key, subkey = jax.random.split(key)`.

## Prerequisites

- macOS 12.0 or later
- Apple Silicon Mac (M1, M2, or M3 family)
- Python 3.9 or later
- pip

## Time & risk

- **Duration**: ~15 minutes
- **Risk level**: Low — small packages, no model downloads, no GPU state persistence
- **Note**: `jax-metal` is experimental. Not all JAX operations are implemented. Check the jax-metal GitHub issues for known unsupported ops.

<!-- tab: Install -->
## Step 1: Install JAX and the Metal backend

The `jax-metal` package version must exactly match the `jax` version it was built for. Installing them together in one pip call ensures pip resolves compatible versions:

```bash
# Create an isolated environment
python3 -m venv ~/.venvs/jax-metal
source ~/.venvs/jax-metal/bin/activate

# Install JAX CPU version first, then add the Metal plugin
# jax-metal installs as an XLA backend plugin — it extends jax, not replaces it
pip install jax
pip install jax-metal

# Verify the installed versions match
python -c "
import jax, jaxlib
print(f'jax:     {jax.__version__}')
print(f'jaxlib:  {jaxlib.__version__}')
"

# Check the jax-metal PyPI page for compatible version pairs if you hit errors:
# https://pypi.org/project/jax-metal/
```

If you see version mismatch errors, pin both packages to a known-good pair:

```bash
# Example of pinning to a specific compatible pair
pip install "jax==0.4.25" "jax-metal==0.1.0"
```

## Step 2: Verify the Metal GPU device is visible

A successful `jax-metal` install causes JAX to discover the Metal GPU as an available device:

```bash
python -c "
import jax

# List all available devices
devices = jax.devices()
print(f'Available devices: {devices}')
# Expected: [METAL(id=0)]
# If you see: [CpuDevice(id=0)] — jax-metal is not installed or not loading

# Default device should be Metal
print(f'Default device: {jax.default_device()}')

# Check backend name
print(f'Backend: {jax.default_backend()}')
# Expected: metal
"
```

If you see `[CpuDevice(id=0)]` instead of `[METAL(id=0)]`, `jax-metal` is not loading. Check that both packages are installed in the same venv and that the versions are compatible.

## Step 3: Run a JIT-compiled function and observe compilation caching

JIT compilation is JAX's central performance mechanism. The first call is slow (compilation), subsequent calls are fast (cached):

```python
import jax
import jax.numpy as jnp
import time

@jax.jit
def matmul_relu(a, b):
    """JIT-compiled matrix multiply followed by ReLU activation."""
    return jnp.maximum(0, jnp.dot(a, b))  # ReLU: max(0, x)

key = jax.random.PRNGKey(0)
a = jax.random.normal(key, (1000, 1000))
b = jax.random.normal(key, (1000, 1000))

# First call: JAX traces the function, compiles to Metal, then runs
# This will take several seconds — that's compilation, not slow execution
start = time.perf_counter()
result = matmul_relu(a, b)
result.block_until_ready()  # JAX is async — block until GPU finishes
print(f"First call (compile + run): {time.perf_counter()-start:.3f}s")

# Second call: compiled version is cached, runs at full GPU speed
start = time.perf_counter()
result = matmul_relu(a, b)
result.block_until_ready()
print(f"Second call (cached):       {time.perf_counter()-start:.3f}s")

# The ratio between first and second call shows compilation overhead
# Typical: first call 5-30s, second call <0.1s
```

## Step 4: Test automatic differentiation

Automatic differentiation is JAX's killer feature for ML research. `jax.grad` computes exact gradients of any JAX function:

```python
import jax
import jax.numpy as jnp

def quadratic(x):
    """f(x) = x^2 + 2x + 1. Derivative: f'(x) = 2x + 2."""
    return x**2 + 2*x + 1

# grad returns a function that computes the gradient
grad_fn = jax.grad(quadratic)

x = jnp.array(3.0)
print(f"f(3)  = {quadratic(x)}")   # 16.0
print(f"f'(3) = {grad_fn(x)}")     # 8.0 (= 2*3 + 2)

# grad works on functions of arrays too (computes Jacobian-vector products)
def loss(params):
    """Simple MSE loss for a linear model y = w*x + b."""
    w, b = params["w"], params["b"]
    x = jnp.array([1.0, 2.0, 3.0])
    y_true = jnp.array([2.0, 4.0, 6.0])
    y_pred = w * x + b
    return jnp.mean((y_pred - y_true)**2)

params = {"w": jnp.array(1.5), "b": jnp.array(0.0)}
grads = jax.grad(loss)(params)
print(f"dL/dw = {grads['w']}")  # Gradient of loss w.r.t. weight
print(f"dL/db = {grads['b']}")  # Gradient of loss w.r.t. bias
```

<!-- tab: Examples -->
## Step 1: Neural network from scratch with JAX

JAX provides the building blocks to implement neural networks without a framework. This shows how `grad` enables training:

```python
import jax
import jax.numpy as jnp
from functools import partial

# Network: input(784) -> hidden(256) -> output(10) (MNIST-sized)
def init_params(key, layer_sizes):
    """Initialize weights with Xavier/Glorot initialization."""
    params = []
    for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
        key, subkey = jax.random.split(key)   # Always split — never reuse keys
        W = jax.random.normal(subkey, (in_size, out_size)) * jnp.sqrt(2.0 / in_size)
        b = jnp.zeros(out_size)
        params.append({"W": W, "b": b})
    return params

def forward(params, x):
    """Forward pass through all layers."""
    for i, layer in enumerate(params[:-1]):
        x = jnp.dot(x, layer["W"]) + layer["b"]
        x = jnp.maximum(0, x)  # ReLU activation
    # Final layer: no activation (raw logits)
    last = params[-1]
    return jnp.dot(x, last["W"]) + last["b"]

def cross_entropy_loss(params, x_batch, y_batch):
    """Cross-entropy loss over a batch."""
    logits = forward(params, x_batch)
    # Stable softmax: subtract max for numerical stability
    log_probs = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    return -jnp.mean(log_probs[jnp.arange(len(y_batch)), y_batch])

# Initialize and do one gradient step
key = jax.random.PRNGKey(42)
params = init_params(key, [784, 256, 10])

# JIT-compile the gradient function for GPU execution
grad_loss = jax.jit(jax.grad(cross_entropy_loss))

# Fake batch for demonstration
key, subkey = jax.random.split(key)
x_batch = jax.random.normal(subkey, (32, 784))
y_batch = jnp.zeros(32, dtype=jnp.int32)

grads = grad_loss(params, x_batch, y_batch)
learning_rate = 0.01

# SGD update — JAX uses pytree traversal for nested dict params
params_updated = jax.tree_util.tree_map(
    lambda p, g: p - learning_rate * g,
    params, grads
)
print("One gradient step complete")
```

## Step 2: Vectorized computation with vmap

`vmap` eliminates explicit batch dimensions from your code. Write a function for a single example, then vectorize it automatically:

```python
import jax
import jax.numpy as jnp

def dot_product(a, b):
    """Dot product of two 1D vectors."""
    return jnp.dot(a, b)  # Works for 1D arrays only

# Without vmap: must manually handle batch dimension
def batch_dot_manual(a_batch, b_batch):
    return jnp.sum(a_batch * b_batch, axis=-1)

# With vmap: JAX maps dot_product over the batch dimension automatically
# vmap(dot_product) generates ONE GPU kernel that processes the entire batch
# A Python for loop would generate N separate GPU kernel calls
batch_dot_vmapped = jax.vmap(dot_product)

key = jax.random.PRNGKey(0)
a_batch = jax.random.normal(key, (1000, 512))  # 1000 vectors of dimension 512
b_batch = jax.random.normal(key, (1000, 512))

result = batch_dot_vmapped(a_batch, b_batch)
print(f"Result shape: {result.shape}")  # (1000,) — one scalar per pair

# vmap + jit: vectorize AND compile
fast_batch_dot = jax.jit(jax.vmap(dot_product))
result = fast_batch_dot(a_batch, b_batch)
result.block_until_ready()
print(f"JIT + vmap result shape: {result.shape}")

# vmap over arbitrary axes — in_axes specifies which axis to map over
# This maps over axis 1 of a, axis 0 of b:
result_custom = jax.vmap(dot_product, in_axes=(1, 0))(a_batch.T, b_batch)
```

## Step 3: MuJoCo MJX setup (JAX is a prerequisite)

MuJoCo MJX is the JAX-based physics simulation backend that achieves massively parallel rollouts. JAX on Metal is a prerequisite:

```bash
# Install MuJoCo with MJX support
pip install mujoco mujoco-mjx

# Verify MJX can find JAX
python -c "
import mujoco
import mujoco.mjx as mjx
import jax
print(f'MuJoCo: {mujoco.__version__}')
print(f'JAX devices: {jax.devices()}')
print('MJX is ready — see the MuJoCo MJX playbook for full usage')
"
```

MJX runs physics simulation as a JAX computation, meaning the entire simulation step is JIT-compiled and can be vectorized with `vmap` to run thousands of parallel environments simultaneously.

## Step 4: Optax for gradient-based optimization

Optax provides JAX-native optimizers (Adam, SGD, AdaGrad) that work seamlessly with `jax.grad`:

```bash
pip install optax
```

```python
import jax
import jax.numpy as jnp
import optax

# Define a simple quadratic to minimize: f(x) = x^2 + y^2
def loss(params):
    return params["x"]**2 + params["y"]**2

# Adam optimizer
optimizer = optax.adam(learning_rate=0.1)
params = {"x": jnp.array(5.0), "y": jnp.array(-3.0)}
opt_state = optimizer.init(params)

@jax.jit
def step(params, opt_state):
    grads = jax.grad(loss)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss(new_params)

# Run 50 optimization steps
for i in range(50):
    params, opt_state, loss_val = step(params, opt_state)
    if i % 10 == 0:
        print(f"Step {i:3d}: loss={loss_val:.6f}  x={params['x']:.4f}  y={params['y']:.4f}")

# Should converge toward x≈0, y≈0
```

<!-- tab: Troubleshooting -->
## Common issues

| Symptom | Cause | Fix |
|---|---|---|
| `NotImplementedError: unimplemented primitive: ...` | Op not yet in jax-metal | Run on CPU: `with jax.default_device(jax.devices('cpu')[0]): result = op(x)` |
| Version mismatch error on import | jax and jax-metal versions incompatible | `pip install jax==X.Y.Z jax-metal==A.B.C` using the paired versions from PyPI |
| `jax.devices()` shows `[CpuDevice(id=0)]` | jax-metal not installed or failed to load | Verify both packages in same venv: `pip show jax jax-metal` |
| `ConcretizationTypeError` inside `jit` | Inspecting concrete values during JIT tracing | Move the concrete value check outside `@jax.jit`, or use `jax.lax.cond` |
| First call takes 30+ seconds | JIT compilation — expected behavior | This is normal. Only the first call per function+shape combination compiles. |
| `RuntimeError: Out of memory` | Array or batch too large for Metal memory | Reduce batch size or array dimensions; call `jax.clear_caches()` between runs |
| `UnexpectedTracerError: Encountered a Jax Tracer` | In-place mutation inside `jit` | Replace `x[i] = val` with `x = x.at[i].set(val)` — JAX's immutable update syntax |

## Unsupported ops: CPU fallback pattern

When `jax-metal` doesn't support an operation, fall back to CPU for just that operation:

```python
import jax
import jax.numpy as jnp

cpu = jax.devices("cpu")[0]
gpu = jax.devices("metal")[0]  # or jax.devices()[0]

def compute_with_fallback(x):
    # Run most operations on GPU
    x_gpu = jax.device_put(x, gpu)
    x_transformed = jax.jit(jnp.fft.fft)(x_gpu)  # May fail on Metal

    # Fall back to CPU for unsupported op
    x_cpu = jax.device_put(x_transformed, cpu)
    result_cpu = jnp.linalg.eig(x_cpu)  # eig may not be in jax-metal yet

    # Move result back to GPU for further processing
    return jax.device_put(result_cpu, gpu)
```

## JAX in-place mutation: the `.at[].set()` pattern

This is the most common JAX mistake when coming from NumPy:

```python
import jax.numpy as jnp

# WRONG — will raise UnexpectedTracerError inside jit
x = jnp.zeros(5)
x[2] = 1.0  # Not allowed in JAX

# CORRECT — functional update creates a new array
x = jnp.zeros(5)
x = x.at[2].set(1.0)  # Returns new array with index 2 set to 1.0
print(x)  # [0. 0. 1. 0. 0.]

# Other .at[] operations:
x = x.at[1].add(5.0)    # Equivalent to x[1] += 5.0
x = x.at[3].mul(2.0)    # Equivalent to x[3] *= 2.0
x = x.at[0:2].set(9.0)  # Slice update
```

## Random key management

JAX's functional random number system is a frequent source of bugs for NumPy users:

```python
import jax
import jax.numpy as jnp

# WRONG — reusing a key produces the same numbers every time
key = jax.random.PRNGKey(0)
a = jax.random.normal(key, (3,))
b = jax.random.normal(key, (3,))  # Same as a!
print(jnp.allclose(a, b))  # True — both use the same key

# CORRECT — split the key before each use
key = jax.random.PRNGKey(0)
key, subkey1 = jax.random.split(key)   # key is now "spent", use subkey1 for sampling
a = jax.random.normal(subkey1, (3,))

key, subkey2 = jax.random.split(key)   # key advances, use subkey2 for next sample
b = jax.random.normal(subkey2, (3,))

print(jnp.allclose(a, b))  # False — different values
```
