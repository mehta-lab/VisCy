# %% imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import cv2
from iohub import open_ome_zarr
from iohub.ngff.nodes import TransformationMeta
from skimage.morphology import remove_small_objects

from utils.image_utils import normalize_image

# %% paths
input_zarr_path = Path("input_data.zarr")
output_zarr_path = Path("segmentation_puncta.zarr")

# Name of the organelle fluorescence channel within the zarr store
organelle_channel = "raw GFP EX488 EM525-45"

# Segmentation parameters
min_puncta_size = 3       # minimum connected-component area in pixels (2-D)
median_kernel_size = 5    # kernel size for median blur (background estimation)
threshold_percentile = 99.9  # intensity percentile for binarisation after subtraction

# Output channel name written into the segmentation zarr
OUTPUT_CHANNEL_NAME = "Organelle_mask"


# %% helper – segment a single 2-D frame
def segment_frame_2d(frame: np.ndarray) -> np.ndarray:
    """Segment organelle puncta in a single 2-D image.

    Pipeline:
    1. Normalise to uint8.
    2. Median blur (background estimation).
    3. Subtract median-blurred image from the normalised image.
    4. Threshold at ``threshold_percentile``-th percentile.
    5. Remove connected components smaller than ``min_puncta_size`` pixels.

    Parameters
    ----------
    frame:
        2-D numpy array (Y, X) of any numeric dtype.

    Returns
    -------
    np.ndarray
        Boolean binary mask of shape (Y, X).
    """
    # Normalise to [0, 255] uint8
    norm = normalize_image(frame)
    uint8_frame = (norm * 255).clip(0, 255).astype(np.uint8)

    # Median blur for background estimation
    blurred = cv2.medianBlur(uint8_frame, median_kernel_size)

    # Residual (hot-pixel / puncta signal)
    residual = uint8_frame.astype(np.float32) - blurred.astype(np.float32)
    residual = np.clip(residual, 0, None)

    # Threshold
    thresh_val = np.percentile(residual, threshold_percentile)
    binary = residual > thresh_val

    # Remove small objects
    binary = remove_small_objects(binary, min_size=min_puncta_size)

    return binary.astype(np.uint8)


# %% open input zarr and set up output zarr
with open_ome_zarr(input_zarr_path, mode="r") as plate:
    # Collect all well/position paths and their shapes for output initialisation
    positions_info = []
    for well_name, well in plate.wells():
        for pos_name, pos in well.positions():
            img_data = pos["0"]
            T, C, Z, Y, X = img_data.shape
            channel_names = pos.channel_names
            positions_info.append(
                {
                    "well_name": well_name,
                    "pos_name": pos_name,
                    "T": T, "Z": Z, "Y": Y, "X": X,
                    "channel_names": channel_names,
                    "axes": pos.metadata.axes if hasattr(pos.metadata, "axes") else None,
                    "transforms": pos.metadata.coordinateTransformations
                    if hasattr(pos.metadata, "coordinateTransformations") else None,
                }
            )

# Open output zarr for writing
with open_ome_zarr(
    output_zarr_path,
    layout="hcs",
    mode="w",
    channel_names=[OUTPUT_CHANNEL_NAME],
) as out_plate:

    with open_ome_zarr(input_zarr_path, mode="r") as plate:
        for info in positions_info:
            well_name = info["well_name"]
            pos_name = info["pos_name"]
            T = info["T"]
            Z = info["Z"]
            Y = info["Y"]
            X = info["X"]

            in_pos = plate[f"{well_name}/{pos_name}"]
            img_data = in_pos["0"]
            channel_names = in_pos.channel_names

            if organelle_channel not in channel_names:
                print(f"  [WARN] Channel '{organelle_channel}' not found in "
                      f"{well_name}/{pos_name}. Skipping.")
                continue

            org_ch_idx = channel_names.index(organelle_channel)

            # Allocate output mask array: T, 1 (Organelle_mask), Z, Y, X
            mask_volume = np.zeros((T, 1, Z, Y, X), dtype=np.uint8)

            for t in range(T):
                for z in range(Z):
                    frame = np.array(img_data[t, org_ch_idx, z, :, :])
                    mask_volume[t, 0, z, :, :] = segment_frame_2d(frame)

            # Create the position in the output store and write
            out_well = out_plate.require_well(well_name)
            out_pos = out_well.require_position(pos_name)
            out_pos.create_zeros(
                name="0",
                shape=(T, 1, Z, Y, X),
                dtype=np.uint8,
                chunks=(1, 1, 1, Y, X),
            )
            out_pos["0"][:] = mask_volume

            print(f"  Segmented {well_name}/{pos_name}: {mask_volume.sum()} positive pixels "
                  f"across T={T}, Z={Z}")

print(f"\nSegmentation complete. Output written to {output_zarr_path}")
