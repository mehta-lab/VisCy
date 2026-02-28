# Pseudotime Remodeling Analysis

Measure organelle remodeling timing relative to viral infection onset using lineage-aware alignment and multiple signal extraction methods.

## Overview

This directory contains shared library modules and analysis scripts that follow a common pipeline:

```
alignment → signal extraction → aggregation → metrics → plotting
```

### Library Modules

| Module | Description |
|--------|-------------|
| `alignment.py` | Lineage detection, FOV/track filtering, T_perturb assignment |
| `signals.py` | Signal extraction: annotation binary, classifier prediction, embedding distance |
| `metrics.py` | Population aggregation, onset/T50/peak detection, per-track timing, statistical tests |
| `plotting.py` | Response curves, per-track heatmaps, timing distributions, onset comparison |

### Analysis Scripts

Each script runs the full pipeline with a different signal source. They are Jupyter-compatible (`# %%` cell markers) and designed for HPC execution.

| Script | Signal Source | Requires |
|--------|--------------|----------|
| `annotation_remodeling.py` | Human annotations (`organelle_state` column) | Tracking CSV + annotation CSV |
| `prediction_remodeling.py` | Classifier predictions (`predicted_organelle_state` in AnnData) | Tracking CSV + predicted AnnData zarr |
| `embedding_distance.py` | Cosine distance from baseline embeddings | Tracking CSV + embedding AnnData zarr |

## Prerequisites

Install DynaCLR with the eval extras and statsmodels:

```bash
cd applications/dynaclr
uv pip install -e ".[eval]" statsmodels
```

## Running Tests

Unit tests cover all four library modules using synthetic data (no HPC paths required):

```bash
cd applications/dynaclr
uv run pytest tests/test_pseudotime.py -v
```

### Test Structure

| Test Class | Tests | Module Covered |
|------------|-------|----------------|
| `TestAlignment` | 7 | `alignment.py` — lineage detection, FOV filtering, T_perturb assignment |
| `TestSignals` | 5 | `signals.py` — annotation/prediction/embedding-distance signal extraction |
| `TestMetrics` | 8 | `metrics.py` — population aggregation, onset/T50/peak, track timing, stats |
| `TestPlotting` | 4 | `plotting.py` — file output (pdf+png) and Figure return for all plot types |

### Synthetic Data

Tests use a self-contained tracking DataFrame with:
- **C/2/000**: 3 tracks with parent-child lineage, infected at t=5
- **C/2/001**: 1 orphan track, infected at t=7
- **B/1/000**: 2 control tracks (no infection)

Plus a matching AnnData with 16-dim random embeddings and classifier predictions.

## Pipeline Details

### 1. Alignment

Tracks are filtered by FOV pattern and minimum length, then aligned to infection onset (T_perturb). Lineage-aware logic ensures all tracks in a parent-child lineage share the same T_perturb.

```python
from pseudotime.alignment import align_tracks

aligned_df = align_tracks(
    tracking_df,
    frame_interval_minutes=30.0,
    fov_pattern="C/2",
    min_track_timepoints=3,
)
# Adds columns: t_perturb, t_relative_minutes
```

### 2. Signal Extraction

Three modes producing a common `signal` column:

```python
from pseudotime.signals import (
    extract_annotation_signal,
    extract_prediction_signal,
    extract_embedding_distance,
)

# Binary from annotations
df = extract_annotation_signal(aligned_df, state_col="organelle_state")

# Binary or continuous from classifier predictions
df = extract_prediction_signal(adata, aligned_df, task="organelle_state")

# Cosine distance from baseline embeddings
df = extract_embedding_distance(adata, aligned_df, baseline_method="per_track")
```

### 3. Aggregation and Metrics

```python
from pseudotime.metrics import aggregate_population, find_onset_time

time_bins = np.arange(-600, 901, 30)
pop_df = aggregate_population(df, time_bins, signal_type="fraction")
onset, threshold, bl_mean, bl_std = find_onset_time(pop_df)
```

### 4. Plotting

All plot functions save pdf+png and return the matplotlib Figure:

```python
from pseudotime.plotting import plot_response_curves

fig = plot_response_curves(
    organelle_curves={"SEC61": pop_df},
    organelle_configs={"SEC61": {"label": "SEC61", "color": "#1f77b4"}},
    output_dir=Path("figures/"),
)
```
