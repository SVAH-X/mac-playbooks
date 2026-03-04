---
slug: mlx-data-science
title: "MLX + Accelerate Data Science"
time: "20 min"
color: green
desc: "GPU-accelerated numerical computing on Apple Silicon"
tags: [data science, mlx]
spark: "CUDA-X Data Science"
category: data-science
featured: false
whatsNew: false
---

<!-- tab: Overview -->
## Basic idea

Apple Silicon's unified memory architecture means that CPU and GPU share the same physical RAM pool. There is no PCIe bus copy — the same memory address that the CPU reads from is the same address the Metal GPU reads from. This eliminates the major bottleneck of CUDA GPU programming, where copying data from CPU to GPU (`cudaMemcpy`) often takes longer than the computation itself.

MLX exploits this with lazy evaluation: operations are queued and fused into a computation graph before being dispatched as a single Metal kernel, similar to how JAX traces computation. For data science workloads, this means large matrix operations (PCA, SVD, correlation matrices, linear algebra) run on the Metal GPU without any explicit data movement.

NumPy and scikit-learn on macOS automatically use Apple's Accelerate framework for BLAS operations (matrix multiply, dot products, decompositions). This gives CPU-level acceleration that often outperforms naive CUDA BLAS on the same class of matrix sizes.

## What you'll accomplish

A working GPU-accelerated data science environment with benchmarked performance: MLX for Metal GPU arrays, polars for fast Rust-based DataFrames, and scikit-learn automatically accelerated via Accelerate/BLAS — with timing comparisons showing where each library wins.

## What to know before starting

- **BLAS**: Basic Linear Algebra Subprograms — a standard interface for matrix operations (GEMM, dot, etc.). Apple's Accelerate framework provides an ARM-optimized BLAS implementation that NumPy, SciPy, and scikit-learn link against automatically on macOS. You don't configure this — it just works.
- **Lazy evaluation**: MLX does not execute operations when you call `mx.array([1,2,3]) + mx.array([4,5,6])`. It builds a computation graph. Nothing runs until you call `mx.eval()` or print the result. Forgetting `mx.eval()` in benchmarks will give misleading timing results.
- **Unified memory and GPU ops**: Unlike CUDA, you do not call `.to(device)` or `cudaMemcpy`. MLX arrays are already in unified memory. The decision to run on GPU vs CPU happens inside MLX based on operation type and array size.
- **Vectorized operations vs Python loops**: A Python loop over 10,000 elements calls the Python interpreter 10,000 times. A vectorized operation calls one C/Metal function once. For numerical work, always express operations as array operations rather than element-wise Python code.

## Prerequisites

- macOS 14.0 or later
- Apple Silicon Mac
- Python 3.10 or later
- pip

## Time & risk

- **Duration**: ~20 minutes
- **Risk level**: None — all packages are read-only installs, no model downloads, no GPU state changes

<!-- tab: Setup -->
## Step 1: Create a dedicated virtual environment

NumPy version conflicts are common when data science packages are installed into a shared environment. Different packages require different NumPy ABI versions, and pip resolves these incorrectly about half the time. A dedicated venv avoids this entirely.

```bash
# Create environment specifically for data science work
python3 -m venv ~/.venvs/ds-mlx
source ~/.venvs/ds-mlx/bin/activate

# Confirm the right Python is active
which python  # ~/.venvs/ds-mlx/bin/python
python --version  # Python 3.10.x or later
```

## Step 2: Install the full data science stack

Each package serves a specific purpose in the pipeline:

```bash
# Install all packages in one call — pip resolves compatible versions together
pip install mlx numpy pandas polars scipy scikit-learn matplotlib jupyter

# What each package does:
# mlx          — Apple's GPU array library with lazy evaluation on Metal
# numpy        — CPU arrays; links against Accelerate BLAS automatically
# pandas       — Row-based DataFrames (familiar API, less performant than polars)
# polars       — Column-based DataFrames written in Rust; ~5-10x faster than pandas
# scipy        — Scientific computing (stats, optimization, signal processing)
# scikit-learn — ML algorithms, all using Accelerate BLAS internally
# matplotlib   — Plotting and visualization
# jupyter      — Interactive notebooks
```

## Step 3: Verify GPU acceleration is active

Confirm that MLX is targeting the Metal GPU and that NumPy is linked against Accelerate:

```python
import mlx.core as mx
import numpy as np

# MLX should show Device(gpu, 0) — if it shows cpu, something is wrong
print(f"MLX default device: {mx.default_device()}")

# Verify NumPy is using Accelerate (not a generic BLAS)
np.show_config()
# Look for "accelerate" or "vecLib" in the blas_libraries line
# Example: blas_libraries = ['Accelerate']

# Quick MLX GPU smoke test
a = mx.random.normal((1000, 1000))
b = mx.random.normal((1000, 1000))
c = a @ b          # Queues the operation — not executed yet (lazy evaluation)
mx.eval(c)         # Now it runs on Metal GPU
print(f"MLX matmul result shape: {c.shape}")  # (1000, 1000)
print("GPU acceleration: OK")
```

## Step 4: Set up Jupyter with the correct kernel

If you don't register the venv with Jupyter, notebook cells will use the wrong Python interpreter:

```bash
# Install the venv as a named Jupyter kernel
pip install ipykernel
python -m ipykernel install --user --name ds-mlx --display-name "Data Science (MLX)"

# Start Jupyter
jupyter notebook

# In the notebook: Kernel > Change Kernel > "Data Science (MLX)"
# Verify the right environment is active inside the notebook:
# import sys; print(sys.executable)  # Should show ~/.venvs/ds-mlx/bin/python
```

<!-- tab: Examples -->
## Step 1: MLX array basics and lazy evaluation

Understanding when MLX actually executes operations is essential for correct code and accurate benchmarks:

```python
import mlx.core as mx
import numpy as np
import time

# Array creation — these are queued, not executed
a = mx.array([1.0, 2.0, 3.0])
b = mx.array([4.0, 5.0, 6.0])
c = a + b  # Still not executed — this is just a graph node

# mx.eval() forces execution on the Metal GPU
mx.eval(c)
print(c)  # array([5, 7, 9], dtype=float32)

# Printing also forces evaluation (calls mx.eval internally)
print(a @ b)  # 32.0

# Interop with NumPy — conversion copies data but stays in unified memory
np_array = np.array(c)   # MLX -> NumPy
mx_array = mx.array(np_array)  # NumPy -> MLX
print(type(np_array))    # <class 'numpy.ndarray'>
print(type(mx_array))    # <class 'mlx.core.array'>

# When to call mx.eval():
# 1. Before timing a block of MLX code (otherwise you time the graph build, not execution)
# 2. Before converting to NumPy
# 3. When debugging (to see actual values rather than lazy references)
```

## Step 2: GPU-accelerated matrix operations with benchmarks

The Metal GPU wins on large matrix operations. Smaller matrices have too little work to amortize the Metal kernel launch overhead:

```python
import mlx.core as mx
import numpy as np
import time

def benchmark_matmul(n: int):
    """Compare NumPy (CPU+Accelerate) vs MLX (Metal GPU) for n×n matrix multiply."""
    np_a = np.random.randn(n, n).astype(np.float32)
    np_b = np.random.randn(n, n).astype(np.float32)

    # NumPy/Accelerate timing
    start = time.perf_counter()
    _ = np_a @ np_b
    cpu_time = time.perf_counter() - start

    # MLX/Metal timing — must call mx.eval() to get actual execution time
    mx_a = mx.array(np_a)
    mx_b = mx.array(np_b)
    mx.eval(mx_a, mx_b)  # Ensure data transfer is complete before timing

    start = time.perf_counter()
    result = mx_a @ mx_b
    mx.eval(result)      # Force GPU execution before stopping timer
    gpu_time = time.perf_counter() - start

    print(f"n={n:6d}  NumPy/CPU: {cpu_time*1000:.1f}ms  MLX/GPU: {gpu_time*1000:.1f}ms  "
          f"Speedup: {cpu_time/gpu_time:.1f}x")

# Small matrices: CPU wins (Metal launch overhead dominates)
benchmark_matmul(100)    # n=100: NumPy likely faster
benchmark_matmul(500)    # n=500: roughly equal
benchmark_matmul(2000)   # n=2000: MLX wins
benchmark_matmul(5000)   # n=5000: MLX significantly faster
```

## Step 3: GPU-accelerated PCA with MLX

scikit-learn's PCA uses Accelerate BLAS for SVD, which is fast. For even larger datasets, MLX can compute the SVD directly on the GPU:

```python
import mlx.core as mx
import mlx.linalg as mxl
import numpy as np
from sklearn.decomposition import PCA
import time

# Generate a large dataset
n_samples, n_features = 50_000, 200
data = np.random.randn(n_samples, n_features).astype(np.float32)

# --- scikit-learn PCA (uses Accelerate BLAS, runs on CPU) ---
start = time.perf_counter()
pca = PCA(n_components=10)
pca.fit(data)
print(f"sklearn PCA: {time.perf_counter() - start:.3f}s")

# --- MLX PCA (runs SVD on Metal GPU) ---
def mlx_pca(X: mx.array, n_components: int) -> mx.array:
    """PCA via SVD on Metal GPU."""
    # Center the data
    mean = X.mean(axis=0)
    X_centered = X - mean

    # Covariance matrix (GPU matmul)
    cov = (X_centered.T @ X_centered) / (X_centered.shape[0] - 1)

    # SVD on GPU
    U, S, Vt = mxl.svd(cov)

    # Project onto top N components
    components = Vt[:n_components]
    return X_centered @ components.T, S[:n_components]

mx_data = mx.array(data)
mx.eval(mx_data)  # Ensure data is resident before timing

start = time.perf_counter()
projected, variance = mlx_pca(mx_data, n_components=10)
mx.eval(projected, variance)
print(f"MLX PCA:    {time.perf_counter() - start:.3f}s")
print(f"Projected shape: {projected.shape}")   # (50000, 10)
```

## Step 4: Polars DataFrame operations

Polars stores data in columnar format (all values of a column are contiguous in memory), which makes aggregations and groupby operations much faster than pandas' row-based storage:

```python
import polars as pl
import time

# Create a synthetic dataset (in real use: pl.read_csv("data.csv"))
n = 1_000_000
df = pl.DataFrame({
    "category": ["A", "B", "C", "D"] * (n // 4),
    "value":    [float(i % 100) for i in range(n)],
    "score":    [float(i * 0.001) for i in range(n)],
})

# Groupby aggregation — Polars executes this in parallel using multiple CPU cores
start = time.perf_counter()
result = df.group_by("category").agg([
    pl.col("value").mean().alias("mean_value"),
    pl.col("value").std().alias("std_value"),
    pl.col("score").sum().alias("total_score"),
    pl.len().alias("count"),
])
print(f"Polars groupby: {(time.perf_counter()-start)*1000:.1f}ms")
print(result)

# Lazy evaluation in Polars (similar to MLX — builds a query plan, then executes)
lazy_result = (
    df.lazy()
    .filter(pl.col("value") > 50)
    .group_by("category")
    .agg(pl.col("score").mean())
    .sort("category")
    .collect()  # Execute the query plan
)
print(lazy_result)
```

## Step 5: Mixed pipeline — polars to numpy to MLX

The real-world pattern: load data with polars, preprocess with numpy/sklearn, compute custom operations with MLX:

```python
import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler
import mlx.core as mx

# Stage 1: Load and filter with Polars (fast, parallel, no data copy)
df = pl.read_csv("your_data.csv")
features = df.select(["feature_1", "feature_2", "feature_3"]).to_numpy()

# Stage 2: Preprocess with scikit-learn (uses Accelerate BLAS)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features).astype(np.float32)

# Stage 3: Custom GPU computation with MLX (Metal)
mx_features = mx.array(features_scaled)
correlation_matrix = mx_features.T @ mx_features / len(mx_features)
mx.eval(correlation_matrix)

print(f"Correlation matrix shape: {correlation_matrix.shape}")
print(f"Diagonal (should be ~variance): {np.array(mx.diagonal(correlation_matrix))}")
```

<!-- tab: Performance -->
## Benchmark: NumPy CPU vs MLX GPU

Run this script to generate your own performance numbers. Results vary significantly by chip generation:

```python
import mlx.core as mx
import numpy as np
import time

def benchmark(label: str, fn, warmup: int = 3, runs: int = 10):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    mean_ms = (sum(times) / len(times)) * 1000
    print(f"  {label:<30} {mean_ms:8.2f} ms")

n = 4096

np_a = np.random.randn(n, n).astype(np.float32)
np_b = np.random.randn(n, n).astype(np.float32)
mx_a = mx.array(np_a)
mx_b = mx.array(np_b)
mx.eval(mx_a, mx_b)

print(f"\nMatrix size: {n}x{n}  (float32)")
print("=" * 50)

# NumPy operations (Accelerate BLAS on CPU)
benchmark("NumPy matmul",       lambda: np.dot(np_a, np_b))
benchmark("NumPy SVD",          lambda: np.linalg.svd(np_a, full_matrices=False))
benchmark("NumPy corr (manual)",lambda: np_a.T @ np_a / n)

print()

# MLX operations (Metal GPU)
benchmark("MLX matmul", lambda: (r := mx_a @ mx_b, mx.eval(r)))
benchmark("MLX SVD",    lambda: (r := mx.linalg.svd(mx_a), mx.eval(*r)))
benchmark("MLX corr",   lambda: (r := mx_a.T @ mx_a / n, mx.eval(r)))
```

## Expected results by chip

| Operation (4096x4096 float32) | M1 Pro | M2 Max | M3 Max |
|---|---|---|---|
| NumPy matmul (Accelerate) | ~180ms | ~120ms | ~90ms |
| MLX matmul (Metal GPU) | ~45ms | ~25ms | ~15ms |
| NumPy SVD | ~800ms | ~600ms | ~450ms |
| MLX SVD | ~200ms | ~120ms | ~80ms |

## When to use MLX vs NumPy+Accelerate

| Scenario | Recommendation |
|---|---|
| Arrays smaller than 256x256 | NumPy — Metal kernel launch overhead dominates |
| Standard ML/stats (PCA, regression, clustering) | scikit-learn — Accelerate BLAS is already optimized |
| Large matrix multiply (>1000x1000) | MLX — Metal GPU wins significantly |
| Custom operations not in NumPy/sklearn | MLX — write it as array ops, run on GPU |
| Interop with pandas/polars pipelines | Convert once with `mx.array(np_array)`, compute in MLX |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `import mlx` fails with "version too old" | macOS 14.0 required | Upgrade macOS or use NumPy only |
| MLX benchmark shows wrong (zero) timing | Forgot `mx.eval()` — lazy eval | Call `mx.eval(result)` before stopping timer |
| `mx.metal.get_active_memory()` shows OOM | Large arrays exceed Metal memory pool | Reduce array size or call `mx.metal.reset_peak_memory()` between benchmarks |
| Converting MLX to NumPy is slow | `.tolist()` forces Python-level conversion | Use `np.array(mlx_array)` directly — stays in unified memory |
| Matplotlib figures blank in Jupyter | Inline backend not set | Add `%matplotlib inline` at top of notebook |
| Polars and pandas type errors in same pipeline | Mixed DataFrame types | Convert explicitly: `pl.from_pandas(df)` or `df.to_pandas()` |
| NumPy not using Accelerate | Built against generic BLAS | Run `np.show_config()` — reinstall numpy with `pip install --force-reinstall numpy` |
