---
slug: scrna-seq
title: "Single-cell RNA Sequencing"
time: "15 min"
color: green
desc: "End-to-end scRNA-seq workflow with scanpy"
tags: [data science, bioinformatics]
spark: "Single-cell RNA Sequencing"
category: data-science
featured: false
whatsNew: false
---

<!-- tab: Overview -->
## Basic idea

Run a complete single-cell RNA sequencing analysis pipeline on your Mac using scanpy, AnnData, and related tools. This workflow is fully cross-platform.

## Prerequisites

- macOS (any version)
- Python 3.9+
- 16 GB+ memory recommended for large datasets

## Time & risk

- **Duration:** 15 minutes setup, analysis time varies
- **Risk level:** None

<!-- tab: Setup -->
## Install

```bash
pip install scanpy anndata leidenalg python-igraph
```

## Download example data

```python
import scanpy as sc

# Download the PBMC 3k dataset (classic benchmark dataset)
adata = sc.datasets.pbmc3k()
print(adata)  # AnnData object with 2700 obs × 32738 vars
```

<!-- tab: Analysis -->
## Standard preprocessing pipeline

```python
import scanpy as sc

# Load data
adata = sc.datasets.pbmc3k()

# Quality control
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# Normalize
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Feature selection
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var.highly_variable]

# Dimensionality reduction
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver="arpack")
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

# Clustering
sc.tl.leiden(adata, resolution=0.5)
sc.tl.umap(adata)
```

<!-- tab: Visualization -->
## Plot UMAP clusters

```python
import scanpy as sc
import matplotlib
matplotlib.use("Agg")  # for headless environments

sc.pl.umap(adata, color="leiden", save="clusters.png")
```

## Marker gene analysis

```python
# Find marker genes for each cluster
sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon")
sc.pl.rank_genes_groups(adata, n_genes=25, save="markers.png")

# Top markers per cluster
import pandas as pd
markers = pd.DataFrame(adata.uns["rank_genes_groups"]["names"]).head(5)
print(markers)
```

## Save and load

```python
# Save
adata.write("pbmc3k_processed.h5ad")

# Load
adata = sc.read_h5ad("pbmc3k_processed.h5ad")
```
