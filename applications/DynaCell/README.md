# DynaCell Application

This directory contains tools for computing and analyzing virtual staining metrics using the DynaCell benchmark datasets.

## Overview

The DynaCell application provides functionality to:
- Compute intensity-based metrics between target and predicted virtual staining images
- Process multiple infection conditions (Mock, DENV) in single function calls
- Support parallel processing for faster computation
- Generate detailed CSV reports with position names and dataset information

## Key Files

- `compute_virtual_staining_metrics.py` - Main script for computing VSCyto3D and CellDiff metrics
- `benchmark.py` - Core functions for metrics computation with parallel processing support
- `example_parallel_usage.py` - Example showing sequential vs parallel processing
- `test_parallel_metrics.py` - Test script for verifying parallel functionality

## Parallel Processing Architecture

### Dataset Structure and Worker Distribution

The DynaCell metrics pipeline processes data at the individual timepoint level, enabling efficient parallel processing:

```
Dataset Structure:
├── Position B/1/000001 (Mock, A549, HIST2H2BE)
│   ├── Sample 0: timepoint 0  <- Worker A processes this
│   ├── Sample 1: timepoint 1  <- Worker B processes this  
│   ├── Sample 2: timepoint 2  <- Worker C processes this
│   └── Sample 3: timepoint 3  <- Worker D processes this
├── Position B/1/000002 (DENV, A549, HIST2H2BE)  
│   ├── Sample 4: timepoint 0  <- Worker A processes this
│   ├── Sample 5: timepoint 1  <- Worker B processes this
│   └── ...
└── Position B/2/000001 (Mock, A549, HIST2H2BE)
    ├── Sample N: timepoint 0   <- Worker C processes this
    └── ...
```

### How Workers Process Data

**Granularity**: Each worker processes individual `(position, timepoint)` combinations

**Distribution**: PyTorch's DataLoader distributes samples across workers in round-robin fashion:
- With `num_workers=4` and 100 samples: Worker 0 gets samples [0,4,8,12...], Worker 1 gets [1,5,9,13...], etc.

**Batch Processing**: With `batch_size=1`, each worker processes exactly one sample at a time for metrics compatibility

**Concurrency Benefits**:
- **I/O Parallelism**: Workers read different zarr files/slices simultaneously
- **CPU Parallelism**: Image processing, transforms, and metrics computation happen in parallel
- **Memory Efficiency**: Each worker only loads one timepoint at a time

### Performance Optimization

This design is particularly effective for DynaCell data because:
- Each timepoint requires significant I/O (loading image slices from zarr)
- Metrics computation is CPU-intensive  
- Different timepoints are independent and can be processed in any order

**Recommended Settings**:
- `num_workers=4-12` for typical HPC setups
- `batch_size=1` (hardcoded for metrics compatibility)
- More workers help when you have many positions/timepoints

## Usage Examples

### Basic Usage (Sequential)
```python
metrics = compute_metrics(
    metrics_module=IntensityMetrics(),
    cell_types=["A549"],
    organelles=["HIST2H2BE"], 
    infection_conditions=["Mock", "DENV"],
    target_database=database,
    target_channel_name="raw Cy5 EX639 EM698-70",
    prediction_database=database,
    prediction_channel_name="nuclei_prediction",
    log_output_dir=output_dir,
    log_name="metrics_example"
)
```

### Parallel Processing
```python
metrics = compute_metrics(
    # ... same parameters as above ...
    num_workers=8,  # Use 8 workers for parallel processing
)
```

### Multiple Conditions
The system supports processing multiple infection conditions in a single call:
```python
infection_conditions=["Mock", "DENV"]  # Processes both conditions together
```

## Output Format

The generated CSV files include:
- **Standard metrics**: SSIM, PSNR, correlation coefficients, etc.
- **Position information**: `position_name` (e.g., "B/1/000001") 
- **Dataset information**: `dataset` name from the database
- **Condition metadata**: `cell_type`, `organelle`, `infection_condition`
- **Temporal information**: `time_idx` for tracking timepoints

## Thread Safety

The system uses `ParallelSafeMetricsLogger` to prevent race conditions when multiple workers write metrics. This logger:
- Collects metrics in memory during processing
- Writes all data atomically to CSV after completion
- Prevents file corruption from concurrent writes

## Database Structure

The system expects database CSV files with columns:
- `Cell type` - Cell line (e.g., "A549", "HEK293T")
- `Organelle` - Target organelle (e.g., "HIST2H2BE") 
- `Infection` - Condition (e.g., "Mock", "DENV")
- `Path` - Path to zarr files
- `Dataset` - Dataset identifier
- Additional metadata columns

The system automatically handles:
- Filtering by multiple conditions using OR logic
- Deduplication by zarr path AND FOV name  
- Metadata preservation through the processing pipeline