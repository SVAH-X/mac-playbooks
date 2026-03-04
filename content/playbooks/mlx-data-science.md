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

Apple's Accelerate framework provides highly optimized BLAS/LAPACK routines, while MLX gives direct Metal GPU access for numerical computing. Together they replace CUDA-X libraries like cuBLAS and cuDNN on Apple Silicon.

## Key libraries

- **MLX** — GPU-accelerated arrays with NumPy-compatible API
- **Accelerate** — automatically used by NumPy, SciPy, scikit-learn on macOS
- **polars** — fast DataFrames (alternative to cuDF)
- **scikit-learn** — uses Accelerate for BLAS operations automatically

## Prerequisites

- macOS 14.0+
- Apple Silicon Mac
- Python 3.10+

<!-- tab: Setup -->
## Install data science stack

```bash
pip install mlx numpy pandas scipy scikit-learn matplotlib jupyter polars
```

## Verify MLX GPU usage

```python
import mlx.core as mx

# Check device
print(mx.default_device())  # Device(gpu, 0)
```

<!-- tab: Examples -->
## MLX matrix operations (runs on Metal GPU)

```python
import mlx.core as mx
import numpy as np

# MLX arrays live in unified memory — CPU and GPU access the same data
a = mx.random.normal((10000, 10000))
b = mx.random.normal((10000, 10000))

# Matrix multiply runs on Metal GPU automatically
c = a @ b
mx.eval(c)  # force evaluation (MLX is lazy)

# Seamless interop with NumPy
np_array = np.array(c)
print(np_array.shape)
```

## scikit-learn with Accelerate

```python
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

# Accelerate BLAS is used automatically for large matrices
X = np.random.randn(10000, 100)
pca = PCA(n_components=10)
X_reduced = pca.fit_transform(X)

kmeans = KMeans(n_clusters=10)
labels = kmeans.fit_predict(X_reduced)
```

## Polars for fast DataFrames

```python
import polars as pl

df = pl.read_csv("large_dataset.csv")
result = df.group_by("category").agg([
    pl.col("value").mean().alias("mean_value"),
    pl.col("value").std().alias("std_value"),
])
print(result)
```

<!-- tab: Performance -->
## Benchmark MLX vs NumPy

```python
import mlx.core as mx
import numpy as np
import time

n = 5000

# NumPy (CPU with Accelerate)
a_np = np.random.randn(n, n).astype(np.float32)
b_np = np.random.randn(n, n).astype(np.float32)

t0 = time.time()
c_np = a_np @ b_np
print(f"NumPy: {time.time() - t0:.3f}s")

# MLX (Metal GPU)
a_mx = mx.array(a_np)
b_mx = mx.array(b_np)

mx.eval(a_mx, b_mx)  # ensure data is on GPU
t0 = time.time()
c_mx = a_mx @ b_mx
mx.eval(c_mx)
print(f"MLX: {time.time() - t0:.3f}s")
```
