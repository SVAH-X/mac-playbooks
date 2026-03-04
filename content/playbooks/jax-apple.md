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

JAX runs on Apple Silicon via the `jax-metal` plugin, enabling GPU-accelerated scientific computing, automatic differentiation, and JIT compilation on Mac. MuJoCo MJX (JAX backend) achieves 650K steps/sec on M3 Max.

## Prerequisites

- macOS 12.0+
- Apple Silicon Mac
- Python 3.9+

## Time & risk

- **Duration:** 15 minutes
- **Risk level:** Low
- **Note:** `jax-metal` is experimental — not all operations are supported

<!-- tab: Install -->
## Install JAX

```bash
pip install jax jaxlib
```

## Install Metal backend (for GPU acceleration)

```bash
pip install jax-metal
```

## Verify GPU device

```python
import jax
print(jax.devices())  # Should show: [METAL(id=0)]
```

<!-- tab: Examples -->
## GPU-accelerated matrix multiply

```python
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (5000, 5000))
result = jnp.dot(x, x.T)
print(f"Result shape: {result.shape}")
```

## JIT compilation

```python
import jax
import jax.numpy as jnp

@jax.jit
def matrix_multiply(a, b):
    return jnp.dot(a, b)

key = jax.random.PRNGKey(0)
a = jax.random.normal(key, (1000, 1000))
b = jax.random.normal(key, (1000, 1000))

# First call compiles, subsequent calls are fast
result = matrix_multiply(a, b)
print(result.shape)
```

## Automatic differentiation

```python
import jax
import jax.numpy as jnp

def loss_fn(params, x, y):
    return jnp.mean((params["w"] @ x + params["b"] - y) ** 2)

grad_fn = jax.grad(loss_fn)
params = {"w": jnp.ones((3, 3)), "b": jnp.zeros(3)}
grads = grad_fn(params, jnp.ones((3, 10)), jnp.zeros((3, 10)))
```

<!-- tab: Troubleshooting -->
## Operation not supported on Metal

Some JAX operations aren't implemented in `jax-metal` yet. You'll see errors like `NotImplementedError`.

Fall back to CPU for unsupported ops:
```python
with jax.default_device(jax.devices("cpu")[0]):
    result = unsupported_op(x)
```

## jax-metal import error

Ensure you installed `jax-metal` in the same environment as `jax`. Try:
```bash
pip install --upgrade jax jax-metal
```

## Out of memory

Reduce array sizes or use smaller batches. Use `jax.clear_caches()` between runs.
