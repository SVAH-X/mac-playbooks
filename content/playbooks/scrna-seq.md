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

Single-cell RNA sequencing (scRNA-seq) measures gene expression in thousands of individual cells simultaneously, revealing cellular heterogeneity that bulk RNA-seq obscures by averaging across cell populations. The standard analysis pipeline moves through these stages: raw count matrix → quality control (remove dead cells and doublets) → normalization (correct for library size) → dimensionality reduction (PCA to 50 components) → nearest-neighbor graph → Leiden clustering → UMAP visualization → marker gene identification. All steps run in Python via `scanpy`, which uses scipy sparse matrices internally and handles 100k+ cells on a standard Mac without running out of memory.

## What you'll accomplish

A complete scRNA-seq analysis of the PBMC 3k dataset — 2,700 peripheral blood mononuclear cells profiled across 32,738 genes — starting from raw UMI counts and finishing with a labeled UMAP showing distinct immune cell populations, identified marker genes per cluster, and a processed `h5ad` file ready for publication figures.

## What to know before starting

- **Count matrix** — Rows are cells, columns are genes, and values are UMI (Unique Molecular Identifier) counts: the number of mRNA transcripts from that gene captured in that cell. The matrix is extremely sparse (~90% zeros for most datasets).
- **AnnData structure** — scanpy's core data object. `.X` holds the count matrix, `.obs` is a DataFrame of cell-level metadata (e.g., cluster assignment, total counts), `.var` is a DataFrame of gene-level metadata (e.g., highly variable flag), `.obsm` stores embeddings like PCA coordinates and UMAP coordinates.
- **Normalization** — Cells captured in a droplet vary in sequencing depth (total counts). Without normalization, a cell with 2× more total counts appears to express every gene 2× more. Library-size normalization scales each cell so its total count equals 10,000, making cells comparable.
- **PCA and UMAP** — We go from 32,738 gene dimensions to 50 PCA components (capturing ~90% of variance) to 2D UMAP (for visualization). PCA is a linear transformation; UMAP is non-linear and better preserves local neighborhood structure. Clustering happens in PCA space, not UMAP space.
- **Leiden clustering** — A community detection algorithm that partitions the k-nearest-neighbor graph of cells into clusters. The `resolution` parameter controls granularity: higher resolution = more, smaller clusters. Leiden is preferred over Louvain because it guarantees well-connected communities.

## Prerequisites

- macOS (any version)
- Python 3.9+
- 16 GB+ RAM for datasets beyond PBMC 3k; PBMC 3k itself needs only ~4 GB

## Time & risk

- **Duration:** 15 minutes setup; full analysis runs in ~5 minutes on PBMC 3k
- **Risk level:** None — reads and writes local files only
- **Rollback:** Delete the Python environment and the `pbmc3k_processed.h5ad` file

<!-- tab: Setup -->
## Step 1: Install the scanpy ecosystem

Each package has a specific role. Install together to resolve dependency versions automatically.

```bash
pip install scanpy anndata leidenalg python-igraph umap-learn matplotlib
# scanpy      — the full scRNA-seq pipeline (wraps anndata, sklearn, scipy)
# anndata     — the AnnData data structure used by scanpy
# leidenalg   — C++ Leiden clustering algorithm (requires python-igraph)
# python-igraph — graph library that leidenalg depends on
# umap-learn  — UMAP dimensionality reduction algorithm
# matplotlib  — scanpy uses it for all plots
```

Verify the installation:

```python
import scanpy as sc
print(sc.__version__)  # should be 1.9+ for full Leiden/UMAP support
```

## Step 2: Configure scanpy settings

Set verbosity and figure defaults before running any analysis so all plots are consistent and reproducible.

```python
import scanpy as sc
import matplotlib
matplotlib.use("Agg")  # use non-interactive backend (saves files instead of popping windows)

sc.settings.verbosity = 3         # 0=errors only, 1=warnings, 2=info, 3=hints (most verbose)
sc.settings.set_figure_params(dpi=100, facecolor="white")
sc.settings.figdir = "./"         # where to save figures

# Set random seeds for reproducibility — UMAP and Leiden both use randomness
import numpy as np
np.random.seed(42)
```

## Step 3: Load the PBMC 3k dataset

The PBMC 3k dataset is a canonical benchmark in single-cell genomics, first published by 10x Genomics. Scanpy hosts it and downloads it automatically (~80 MB) the first time.

```python
# Download from scanpy's dataset repository (cached locally after first run)
adata = sc.datasets.pbmc3k()
print(adata)
# Expected output: AnnData object with n_obs × n_vars = 2700 × 32738
# .obs: ['n_genes'] — only basic metadata at this stage
# .var: ['gene_ids'] — Ensembl gene IDs

print(f"Cells: {adata.n_obs}, Genes: {adata.n_vars}")
print(f"Sparsity: {1 - adata.X.nnz / (adata.n_obs * adata.n_vars):.1%}")
# Expect ~94% sparsity — most gene-cell combinations are zero
```

<!-- tab: Analysis -->
## Step 1: Quality control

Poor-quality cells (damaged, dying, or empty droplets) distort downstream analysis. We flag and remove them using three metrics before doing anything else.

```python
import scanpy as sc
import numpy as np

adata = sc.datasets.pbmc3k()

# Identify mitochondrial genes — their names start with "MT-" in human
adata.var["mt"] = adata.var_names.str.startswith("MT-")

# Compute per-cell QC metrics: total counts, number of genes, % mitochondrial
sc.pp.calculate_qc_metrics(
    adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
)
# After this, adata.obs has columns:
# n_genes_by_counts — number of genes detected in this cell
# total_counts      — total UMI count in this cell
# pct_counts_mt     — fraction of counts from mitochondrial genes

# Visualize the QC distributions before filtering
sc.pl.violin(
    adata, ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    jitter=0.4, multi_panel=True, save="qc_violin.png"
)

# Apply filters:
# n_genes_by_counts > 200: remove empty droplets (few genes = no cell)
# n_genes_by_counts < 2500: remove doublets (too many genes = 2 cells in 1 droplet)
# pct_counts_mt < 5: remove dying cells (mitochondrial mRNA rises as cell degrades)
adata = adata[adata.obs.n_genes_by_counts > 200, :]
adata = adata[adata.obs.n_genes_by_counts < 2500, :]
adata = adata[adata.obs.pct_counts_mt < 5, :]
print(f"After QC: {adata.n_obs} cells remaining")  # expect ~2,600
```

## Step 2: Normalization

After QC, normalize each cell so library size differences don't confound biology. Then apply a log transform to stabilize variance across the expression range.

```python
# Library-size normalization: scale each cell so total counts = 10,000
sc.pp.normalize_total(adata, target_sum=1e4)
# Now each cell has exactly 10,000 total counts (before log transform)
# A gene expressed in 10% of a cell's mRNA will have count = 1,000

# Log1p transform: log(count + 1)
# Compresses the dynamic range (counts range from 0 to 10,000; log range is 0 to ~9.2)
# The +1 handles zeros: log(0+1) = 0, not -infinity
sc.pp.log1p(adata)

# Save the normalized data as the "raw" layer for differential expression later
adata.raw = adata
print("Normalization complete — data is now in log-normalized space")
```

## Step 3: Feature selection — highly variable genes

Not all 32,738 genes are informative. Housekeeping genes (expressed at constant levels in all cells) add noise without signal. We keep only genes that vary substantially across cells.

```python
# Identify the 2,000 most variable genes
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
# min_mean/max_mean: exclude genes expressed at very low or very high constant levels
# min_disp: keep genes with normalized dispersion above this threshold
# Result: adata.var["highly_variable"] column is set to True for selected genes

sc.pl.highly_variable_genes(adata, save="hvg.png")
# The plot shows mean expression vs dispersion — selected genes are in the upper region

# Subset to highly variable genes only (keeps raw copy of all genes in adata.raw)
adata = adata[:, adata.var.highly_variable]
print(f"Retained {adata.n_vars} highly variable genes for downstream analysis")

# Scale each gene to unit variance and zero mean (important for PCA)
sc.pp.scale(adata, max_value=10)
# max_value=10 clips outliers — prevents single extreme cells from dominating PCs
```

## Step 4: PCA

Leiden clustering on 2,000 genes is slow and noisy. PCA compresses the data to 50 linear components that capture most of the variance, making clustering fast and accurate.

```python
sc.tl.pca(adata, svd_solver="arpack", n_comps=50)
# svd_solver="arpack": sparse SVD — much faster than "full" for large matrices
# n_comps=50: compute 50 principal components

# Elbow plot: variance explained per PC — look for the "elbow" where it flattens
sc.pl.pca_variance_ratio(adata, log=True, save="pca_elbow.png")
# The elbow is typically around PC 10-20 for PBMC data
# We use 40 PCs for the neighbor graph (conservative: includes slower PCs)
print(f"Top PC explains {adata.uns['pca']['variance_ratio'][0]:.1%} of variance")
```

## Step 5: Neighborhood graph and Leiden clustering

Build a k-nearest-neighbor graph in PCA space, then detect communities (clusters) in that graph. Cells in the same community have similar expression profiles.

```python
# Build the kNN graph using the first 40 PCs
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
# n_neighbors=10: each cell connects to its 10 nearest neighbors in PC space
# n_pcs=40: use first 40 PCs (safe default; reduce if elbow plot shows earlier flattening)
# This creates adata.obsp["connectivities"] (sparse adjacency matrix)

# Leiden clustering on the kNN graph
sc.tl.leiden(adata, resolution=0.5)
# resolution=0.5: moderate granularity — expect 8-12 clusters for PBMC 3k
# Higher resolution (e.g., 1.0) → more clusters; lower (0.3) → fewer, broader clusters
print(f"Found {adata.obs['leiden'].nunique()} clusters")
print(adata.obs["leiden"].value_counts())  # cluster sizes
```

## Step 6: UMAP embedding

UMAP projects the high-dimensional neighborhood graph into 2D for visualization. Run UMAP only after building the neighbor graph — it uses the graph, not the raw data.

```python
sc.tl.umap(adata)
# Uses the neighbor graph from sc.pp.neighbors()
# Result stored in adata.obsm["X_umap"] — shape (n_cells, 2)
# Each cell gets a (x, y) coordinate in the 2D UMAP space

print("UMAP coordinates shape:", adata.obsm["X_umap"].shape)
# e.g. (2638, 2)
```

<!-- tab: Visualization -->
## Step 1: UMAP colored by cluster

Visualize the UMAP embedding with cells colored by their Leiden cluster assignment. Well-separated blobs indicate distinct cell populations.

```python
import scanpy as sc
import matplotlib
matplotlib.use("Agg")

# Color each cell by its Leiden cluster number
sc.pl.umap(adata, color="leiden", legend_loc="on data", save="umap_clusters.png")
# legend_loc="on data": cluster numbers appear inside each cluster blob
# Tight clusters = strong transcriptional differences
# Overlapping clusters = consider increasing leiden resolution
print("Saved umap_clusters.png")
```

## Step 2: Marker gene discovery

Find genes that are statistically significantly more expressed in each cluster compared to all other cells. These "marker genes" tell you what cell type each cluster represents.

```python
# Wilcoxon rank-sum test: non-parametric, robust to outliers
sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon")
# Tests every gene × every cluster — uses adata.raw (pre-normalization counts)
# Produces: names, scores, pvals_adj (Benjamini-Hochberg), logfoldchanges

# Display top 25 marker genes per cluster
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, save="markers.png")

# Extract top markers into a readable DataFrame
import pandas as pd
marker_df = pd.DataFrame(adata.uns["rank_genes_groups"]["names"]).head(10)
print("Top 10 marker genes per cluster:")
print(marker_df)
# Each column is a cluster; values are gene names ranked by Wilcoxon score
```

## Step 3: Cell type annotation

Map known immune cell marker genes to cluster numbers. This converts cluster "0", "1", etc. into biologically meaningful labels.

```python
# Known PBMC marker genes (from literature)
cell_type_markers = {
    "CD3D": "T cell",          # pan-T cell marker
    "CD19": "B cell",          # pan-B cell marker
    "CD14": "Monocyte",        # classical monocyte
    "FCGR3A": "NK/Monocyte",   # NK cells and non-classical monocytes
    "NKG7": "NK cell",         # natural killer cells
    "FCER1A": "Dendritic cell",# plasmacytoid dendritic cells
    "PPBP": "Platelet",        # megakaryocytes/platelets
}

# Visualize each marker gene on the UMAP — cells with high expression light up
sc.pl.umap(
    adata,
    color=list(cell_type_markers.keys()),
    save="umap_markers.png"
)

# After inspecting which clusters express which markers, annotate manually:
cluster_to_celltype = {
    "0": "CD4 T cells",
    "1": "CD14 Monocytes",
    "2": "B cells",
    "3": "CD8 T cells",
    "4": "NK cells",
    "5": "FCGR3A Monocytes",
    "6": "Dendritic cells",
    "7": "Platelets",
}
adata.obs["cell_type"] = adata.obs["leiden"].map(cluster_to_celltype)
sc.pl.umap(adata, color="cell_type", legend_loc="on data", save="umap_annotated.png")
print("Cell type composition:")
print(adata.obs["cell_type"].value_counts())
```

## Step 4: Save the processed object

The `h5ad` format stores the full AnnData object — matrix, metadata, embeddings, marker gene results — in a single compressed HDF5 file.

```python
# Save all results to disk
output_path = "pbmc3k_processed.h5ad"
adata.write(output_path)
print(f"Saved to {output_path}")
# File contains: normalized counts, PCA, UMAP, Leiden clusters, marker genes, cell type labels

# Load it back later without re-running the pipeline:
# adata = sc.read_h5ad("pbmc3k_processed.h5ad")
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `leidenalg` install fails on Apple Silicon | Binary wheel not available for arm64 | Use `conda install -c conda-forge leidenalg` instead of pip |
| UMAP takes > 10 minutes | Too many neighbors or cells | Reduce `n_neighbors` to 5, or downsample with `sc.pp.subsample(adata, n_obs=10000)` |
| `MemoryError` during PCA | Dense matrix conversion on large dataset | Ensure `adata.X` is sparse: `import scipy.sparse; adata.X = scipy.sparse.csr_matrix(adata.X)` |
| Matplotlib shows blank window | Interactive backend conflict | Add `matplotlib.use("Agg")` before any `import scanpy` |
| Too many or too few clusters | Leiden resolution not tuned | Try `resolution=0.3` (fewer) or `resolution=1.0` (more); re-run from `sc.tl.leiden` |
| All cells in one cluster | Normalization skipped or PCA not run | Ensure you ran `normalize_total`, `log1p`, `scale`, `pca`, then `neighbors` in order |
| Marker genes don't match known biology | QC filters too permissive | Tighten `pct_counts_mt < 5` to `< 3`, remove low-quality cells more aggressively |

### leidenalg on Apple Silicon

The `leidenalg` package requires a C++ extension. The pip wheel sometimes fails on arm64. The reliable fix is conda:

```bash
conda install -c conda-forge leidenalg python-igraph
# Then install the rest via pip inside the same environment:
pip install scanpy umap-learn
```

### Memory-efficient processing for large datasets

For datasets with > 50k cells, avoid dense matrix operations:

```python
import scipy.sparse

# After loading, ensure sparse representation
if not scipy.sparse.issparse(adata.X):
    adata.X = scipy.sparse.csr_matrix(adata.X)

# Use approximate PCA (faster, less memory)
sc.tl.pca(adata, svd_solver="randomized", n_comps=50)

# Reduce neighbor graph size
sc.pp.neighbors(adata, n_neighbors=5, n_pcs=30)  # fewer neighbors = faster
```

### Normalization order matters

Always run QC filtering before normalization. If you normalize first, the per-cell scaling is based on counts from dead cells and doublets, which corrupts the normalization factors for good cells:

```
Correct order: load → QC filter → normalize → log1p → scale → PCA → neighbors → cluster → UMAP
```
