# DynaCLR Classical Sampling

This module implements classical triplet sampling for training DynaCLR models by generating pseudo-tracking data from 2D segmentation masks. It processes segmentation data from an HCS OME-Zarr store and creates corresponding tracking CSV files with the following information:
- Track IDs from segmentation masks
- Centroid coordinates (t, y, x) for each segmented object per time point
- Unique IDs for each object

## Prerequisites
- Input HCS OME-Zarr store containing segmentation masks

## Usage

### 1. Configure Input/Output Paths
Open `create_pseudo_tracks.py` and modify:
```python
# Input path to your segmentation data
input_data_path = "/path/to/your/input.zarr"
# Output path for tracking data
track_data_path = "/path/to/your/output.zarr"
# Channel name for the segmentations
segmentation_channel_name = "Nucl_mask"
# Z-slice to use for 2D tracking
Z_SLICE = 30
```

### 2. Run the Script
```bash
python create_pseudo_tracks.py
```

## Processing Steps
1. Loads segmentation data from input zarr store
2. For each well and position:
   - Processes each timepoint
   - Extracts 2D segmentation at specified z-slice
   - Calculates centroid coordinates for segmented objects (i.e. (y,x))
   - Generates and save the pseudo-tracking data to CSV files
1. Creates a new zarr store with the processed data

## Notes
- Currently only supports 2D segmentation tracking at a single z-slice
- The z-slice index can be modified in the script
- Output CSV files are organized by well and position
- Make sure your zarr stores are properly configured before running the script
