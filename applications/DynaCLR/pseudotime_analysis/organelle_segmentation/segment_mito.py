# %% code for organelle and nuclear segmentation and feature extraction

import os
from pathlib import Path

import napari
import numpy as np
import pandas as pd
from extract_features import (
    extract_features_zyx,
)
from iohub import open_ome_zarr
from matplotlib import pyplot as plt
from segment_organelles import (
    calculate_nellie_sigmas,
    segment_zyx,
)
from skimage.exposure import rescale_intensity
from tqdm import tqdm

os.environ["DISPLAY"] = ":1"
viewer = napari.Viewer()
# %%

# input_path = "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_21_A549_TOMM20_DENV/4-phenotyping/train-test/2024_11_21_A549_TOMM20_DENV.zarr"
input_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/4-phenotyping/train-test/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV.zarr"
)
# input_path = "/hpc/projects/intracellular_dashboard/organelle_box/2025_04_04_organelle_box_Live_dye/4-concatenate/organelle_box_live_dye_TOMM20.zarr"
input_zarr = open_ome_zarr(input_path, mode="r", layout="hcs")
in_chans = input_zarr.channel_names
organelle_channel_name = "GFP EX488 EM525-45"
# Organelle_chan = "4-MultiCam_GFP_Cy5-BSI_Express"

output_root = (
    Path(
        "/home/eduardo.hirata/repos/viscy/applications/pseudotime_analysis/organelle_segmentation/output"
    )
    / input_path.stem
)
output_root.mkdir(parents=True, exist_ok=True)

# %%
# Frangi parameters - WORKING configuration for 2D mitochondria segmentation
frangi_params = {
    "clahe_clip_limit": 0.01,  # Mild contrast enhancement (0.01-0.03 range)
    "sigma_steps": 2,  # Multiple scales to capture size variation
    "auto_optimize_sigma": False,  # Use multi-scale (max across scales)
    "frangi_alpha": 0.5,  # Standard tubular structure sensitivity
    "frangi_beta": 0.5,  # Standard blob rejection
    "threshold_method": "nellie_max",  # CRITICAL: Manual threshold where auto methods fail
    "min_object_size": 10,  # Remove small noise clusters (20-50 pixels)
    "apply_morphology": False,  # Connect fragmented mitochondria
}


position_names = []
for ds, position in input_zarr.positions():
    position_names.append(tuple(ds.split("/")))


for well_id, well_data in input_zarr.wells():
    # print(well_id)
    # well_id, well_data = next(input_zarr.wells())
    well_name, well_no = well_id.split("/")

    if "B/1" not in well_id:
        continue
    # if well_id == 'C/4':
    # print(well_name, well_no)
    for pos_id, pos_data in well_data.positions():
        if pos_id != "000000":
            continue
        scale = (
            pos_data.metadata.multiscales[0]
            .datasets[0]
            .coordinate_transformations[0]
            .scale
        )
        pixel_size_um = scale[-1]  # XY pixel size in micrometers
        z_spacing_um = scale[-3]  # Z spacing in micrometers
        print(f"  Pixel size: {pixel_size_um:.4f} µm, Z spacing: {z_spacing_um:.4f} µm")

        in_data = pos_data.data.numpy()
        if in_data.shape[-3] != 1:
            in_data = np.max(in_data, axis=-3, keepdims=True)
        T, C, Z, Y, X = in_data.shape
        print(f"Input data shape: {in_data.shape} (T={T}, C={C}, Z={Z}, Y={Y}, X={X})")

        # Extract and normalize organelle channel (keep Z dimension)
        organelle_data = in_data[
            :, in_chans.index(organelle_channel_name), :
        ]  # (T, Z, Y, X)
        organelle_data = rescale_intensity(organelle_data, out_range=(0, 1))
        print(f"Organelle data shape after extraction: {organelle_data.shape}")

        # Calculate sigma range - ADJUSTED for your pixel size
        # With 0.1494 µm/pixel, mitochondria (0.3-1.0 µm diameter) = 2-7 pixels diameter
        # Sigma should be ~radius/2, so for diameter 2-7px, sigma = 0.5-1.75 px
        min_radius_um = 0.15  # 300 nm diameter = ~2 pixels
        max_radius_um = 0.6  # 1 µm diameter = ~6.7 pixels
        sigma_range = calculate_nellie_sigmas(
            min_radius_um,
            max_radius_um,
            pixel_size_um,
            num_sigma=frangi_params["sigma_steps"],
        )

        print(f"Using sigma range: {sigma_range[0]:.2f} to {sigma_range[1]:.2f} pixels")

        # Frangi filtering and segmentation
        print(
            f"Computing Frangi segmentation and feature extraction for {well_id}/{pos_id}..."
        )
        frangi_seg_masks = []
        frangi_vesselness_maps = []
        all_features = []

        # FIXME: temporary for testing
        selected_timepoints = np.linspace(0, T - 1, 3).astype(int)
        for t in tqdm(selected_timepoints, desc="Processing timepoints"):
            labels, vesselness, optimal_sigma = segment_zyx(
                organelle_data[t], sigma_range=sigma_range, **frangi_params
            )
            frangi_seg_masks.append(labels[0])
            frangi_vesselness_maps.append(vesselness[0])

            # Extract features from this timepoint
            features_df = extract_features_zyx(
                labels_zyx=labels,
                intensity_zyx=organelle_data[t],
                frangi_zyx=vesselness,
                spacing=(pixel_size_um, pixel_size_um),
                extra_properties=[
                    "aspect_ratio",
                    "circularity",
                    "frangi_intensity",
                    # "texture",
                    # "moments_hu",
                ],
            )

            if not features_df.empty:
                features_df["well_id"] = well_id
                features_df["position_id"] = pos_id
                features_df["timepoint"] = t
                all_features.append(features_df)

        frangi_seg_masks = np.array(frangi_seg_masks)
        frangi_vesselness_maps = np.array(frangi_vesselness_maps)

        # Save combined features
        if all_features:
            combined_features = pd.concat(all_features, ignore_index=True)
            output_csv = output_root / f"features_{well_name}_{well_no}_{pos_id}.csv"
            combined_features.to_csv(output_csv, index=False)
            print(f"  Saved {len(combined_features)} object features to {output_csv}")

        # Convert to output format (T_actual, C=1, Z, Y, X)
        T_actual = frangi_seg_masks.shape[0]
        out_data = frangi_seg_masks[:, :, :].astype(np.uint32)
        print(f"  Processed {T_actual} timepoints, output shape: {out_data.shape}")

        position_key = (well_name, well_no, pos_id)

# %%

viewer.add_image(organelle_data[selected_timepoints, 0])
viewer.add_labels(frangi_seg_masks)


# %%
# Plot mitochondrial dynamics: elongation and fragmentation

if all_features:
    df = combined_features

    # Aggregate features per timepoint
    timepoint_summary = (
        df.groupby("timepoint")
        .agg(
            {
                "label": "count",  # Number of mitochondrial objects
                "area": ["mean", "median", "sum"],  # Size metrics
                "aspect_ratio": ["mean", "median"],  # Elongation metric
                "circularity": ["mean", "median"],  # Shape metric
                "frangi_mean_intensity": ["mean", "median"],  # Tubularity metric
                # "moments_hu_1": ["mean", "median"],  # Shape descriptor
                # "moments_hu_2": ["mean", "median"],  # Shape descriptor
                # "moments_hu_3": ["mean", "median"],  # Shape descriptor
                # "moments_hu_4": ["mean", "median"],  # Shape descriptor
                # "contrast": ["mean", "median"],  # Texture metric
            }
        )
        .reset_index()
    )

    # Flatten column names
    timepoint_summary.columns = [
        "_".join(col).strip("_") for col in timepoint_summary.columns.values
    ]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"Mitochondrial Dynamics: {well_id}/{pos_id}", fontsize=14, fontweight="bold"
    )

    # Plot 1: Number of objects (fragmentation indicator)
    ax = axes[0, 0]
    ax.plot(
        timepoint_summary["timepoint"],
        timepoint_summary["label_count"],
        marker="o",
        linewidth=2,
        markersize=8,
        color="#1f77b4",
    )
    ax.set_xlabel("Timepoint", fontsize=11)
    ax.set_ylabel("Number of Objects", fontsize=11)
    ax.set_title("Fragmentation (Object Count)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Plot 2: Mean area per object
    ax = axes[0, 1]
    ax.plot(
        timepoint_summary["timepoint"],
        timepoint_summary["area_mean"],
        marker="o",
        linewidth=2,
        markersize=8,
        color="#ff7f0e",
        label="Mean",
    )
    ax.plot(
        timepoint_summary["timepoint"],
        timepoint_summary["area_median"],
        marker="s",
        linewidth=2,
        markersize=7,
        color="#d62728",
        label="Median",
        alpha=0.7,
    )
    ax.set_xlabel("Timepoint", fontsize=11)
    ax.set_ylabel("Area (µm²)", fontsize=11)
    ax.set_title("Mitochondrial Size", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Total area (network coverage)
    ax = axes[0, 2]
    ax.plot(
        timepoint_summary["timepoint"],
        timepoint_summary["area_sum"],
        marker="o",
        linewidth=2,
        markersize=8,
        color="#2ca02c",
    )
    ax.set_xlabel("Timepoint", fontsize=11)
    ax.set_ylabel("Total Area (µm²)", fontsize=11)
    ax.set_title("Total Mitochondrial Coverage", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Plot 4: Aspect ratio (elongation)
    ax = axes[1, 0]
    ax.plot(
        timepoint_summary["timepoint"],
        timepoint_summary["aspect_ratio_mean"],
        marker="o",
        linewidth=2,
        markersize=8,
        color="#9467bd",
        label="Mean",
    )
    ax.plot(
        timepoint_summary["timepoint"],
        timepoint_summary["aspect_ratio_median"],
        marker="s",
        linewidth=2,
        markersize=7,
        color="#8c564b",
        label="Median",
        alpha=0.7,
    )
    ax.set_xlabel("Timepoint", fontsize=11)
    ax.set_ylabel("Aspect Ratio", fontsize=11)
    ax.set_title("Elongation (Aspect Ratio)", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Circularity
    ax = axes[1, 1]
    ax.plot(
        timepoint_summary["timepoint"],
        timepoint_summary["circularity_mean"],
        marker="o",
        linewidth=2,
        markersize=8,
        color="#e377c2",
        label="Mean",
    )
    ax.plot(
        timepoint_summary["timepoint"],
        timepoint_summary["circularity_median"],
        marker="s",
        linewidth=2,
        markersize=7,
        color="#7f7f7f",
        label="Median",
        alpha=0.7,
    )
    ax.set_xlabel("Timepoint", fontsize=11)
    ax.set_ylabel("Circularity", fontsize=11)
    ax.set_title("Shape Circularity", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Frangi vesselness (tubularity)
    ax = axes[1, 2]
    ax.plot(
        timepoint_summary["timepoint"],
        timepoint_summary["frangi_mean_intensity_mean"],
        marker="o",
        linewidth=2,
        markersize=8,
        color="#bcbd22",
        label="Mean",
    )
    ax.plot(
        timepoint_summary["timepoint"],
        timepoint_summary["frangi_mean_intensity_median"],
        marker="s",
        linewidth=2,
        markersize=7,
        color="#17becf",
        label="Median",
        alpha=0.7,
    )
    ax.set_xlabel("Timepoint", fontsize=11)
    ax.set_ylabel("Frangi Vesselness", fontsize=11)
    ax.set_title("Tubularity (Frangi)", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_fig = output_root / f"dynamics_{well_name}_{well_no}_{pos_id}.png"
    plt.savefig(output_fig, dpi=300, bbox_inches="tight")
    print(f"  Saved dynamics plot to {output_fig}")

    plt.show()

    # Print summary statistics
    print("\n=== Mitochondrial Dynamics Summary ===")
    print(f"Position: {well_id}/{pos_id}")
    print(f"\nTimepoint range: {selected_timepoints[0]} -> {selected_timepoints[-1]}")
    print("\nFragmentation (Object Count):")
    print(f"  Start: {timepoint_summary['label_count'].iloc[0]:.0f} objects")
    print(f"  End: {timepoint_summary['label_count'].iloc[-1]:.0f} objects")
    print(
        f"  Change: {timepoint_summary['label_count'].iloc[-1] - timepoint_summary['label_count'].iloc[0]:+.0f} ({(timepoint_summary['label_count'].iloc[-1] / timepoint_summary['label_count'].iloc[0] - 1) * 100:+.1f}%)"
    )

    print("\nElongation (Aspect Ratio):")
    print(f"  Start: {timepoint_summary['aspect_ratio_mean'].iloc[0]:.2f}")
    print(f"  End: {timepoint_summary['aspect_ratio_mean'].iloc[-1]:.2f}")
    print(
        f"  Change: {timepoint_summary['aspect_ratio_mean'].iloc[-1] - timepoint_summary['aspect_ratio_mean'].iloc[0]:+.2f} ({(timepoint_summary['aspect_ratio_mean'].iloc[-1] / timepoint_summary['aspect_ratio_mean'].iloc[0] - 1) * 100:+.1f}%)"
    )

    print("\nMean Object Size (Area):")
    print(f"  Start: {timepoint_summary['area_mean'].iloc[0]:.2f} µm²")
    print(f"  End: {timepoint_summary['area_mean'].iloc[-1]:.2f} µm²")
    print(
        f"  Change: {timepoint_summary['area_mean'].iloc[-1] - timepoint_summary['area_mean'].iloc[0]:+.2f} µm² ({(timepoint_summary['area_mean'].iloc[-1] / timepoint_summary['area_mean'].iloc[0] - 1) * 100:+.1f}%)"
    )

# %%
