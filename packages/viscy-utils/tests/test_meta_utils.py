import numpy as np
import pytest
from iohub import open_ome_zarr

from viscy_utils.meta_utils import generate_normalization_metadata

GRID_SPACING = 8


@pytest.fixture(scope="function")
def bimodal_hcs_dataset(tmp_path_factory):
    """HCS dataset with bimodal intensity for Otsu testing."""
    dataset_path = tmp_path_factory.mktemp("bimodal.zarr")
    channel_names = ["Phase", "Fluorescence"]
    zyx_shape = (2, 64, 64)
    with open_ome_zarr(dataset_path, layout="hcs", mode="w", channel_names=channel_names, version="0.5") as plate:
        rng = np.random.default_rng(42)
        for fov_id in ("0", "1"):
            pos = plate.create_position("A", "1", fov_id)
            # Bimodal: background ~0, foreground ~5
            data = np.zeros((1, len(channel_names), *zyx_shape), dtype=np.float32)
            # Phase: uniform noise
            data[:, 0] = rng.random((1, *zyx_shape)).astype(np.float32)
            # Fluorescence: bimodal (BG near 0, FG near 5)
            fluor = rng.random((1, *zyx_shape)).astype(np.float32) * 0.5
            fluor[:, :, 32:, :] += 5.0  # right half is bright
            data[:, 1] = fluor
            pos.create_image("0", data, chunks=(1, 1, *zyx_shape))
    return dataset_path


def test_fov_timepoint_statistics_differ_between_fovs(small_hcs_dataset):
    """Timepoint statistics written to each FOV must reflect that FOV's own data."""
    generate_normalization_metadata(small_hcs_dataset, num_workers=1, grid_spacing=GRID_SPACING)

    with open_ome_zarr(small_hcs_dataset, mode="r") as plate:
        fov_tp_means = {}
        for fov_name, fov in plate.positions():
            tp_stats = fov.zattrs["normalization"]["Phase"]["timepoint_statistics"]
            fov_tp_means[fov_name] = {t: tp_stats[t]["mean"] for t in tp_stats}

        # FOVs were created with offset i*10, so per-timepoint means must differ.
        fov_names = list(fov_tp_means.keys())
        for t in fov_tp_means[fov_names[0]]:
            mean_0 = fov_tp_means[fov_names[0]][t]
            mean_1 = fov_tp_means[fov_names[1]][t]
            assert mean_0 != mean_1, (
                f"FOV {fov_names[0]} and {fov_names[1]} have identical "
                f"timepoint_statistics at t={t} (mean={mean_0}). "
                f"Dataset-level stats were likely copied instead of per-FOV stats."
            )


def test_fov_timepoint_statistics_match_manual_computation(small_hcs_dataset):
    """Per-FOV timepoint statistics must match manually computed values."""
    generate_normalization_metadata(small_hcs_dataset, num_workers=1, grid_spacing=GRID_SPACING)

    with open_ome_zarr(small_hcs_dataset, mode="r") as plate:
        num_timepoints = next(plate.positions())[1]["0"].shape[0]
        for _, fov in plate.positions():
            raw = fov["0"][:]  # (T, C, Z, Y, X)
            norm = fov.zattrs["normalization"]["Phase"]
            for t in range(num_timepoints):
                sampled = raw[t, 0, :, ::GRID_SPACING, ::GRID_SPACING]
                expected_mean = float(np.nanmean(sampled))
                expected_std = float(np.nanstd(sampled))
                actual = norm["timepoint_statistics"][str(t)]
                np.testing.assert_allclose(actual["mean"], expected_mean, rtol=1e-5)
                np.testing.assert_allclose(actual["std"], expected_std, rtol=1e-5)


def test_dataset_timepoint_statistics_on_plate(small_hcs_dataset):
    """Dataset-level timepoint statistics on the plate aggregate across all FOVs."""
    generate_normalization_metadata(small_hcs_dataset, num_workers=1, grid_spacing=GRID_SPACING)

    with open_ome_zarr(small_hcs_dataset, mode="r") as plate:
        plate_tp_stats = plate.zattrs["normalization"]["Phase"]["timepoint_statistics"]
        num_timepoints = next(plate.positions())[1]["0"].shape[0]

        all_fov_data = []
        for _, fov in plate.positions():
            raw = fov["0"][:]
            all_fov_data.append(raw[:, 0, :, ::GRID_SPACING, ::GRID_SPACING])

        for t in range(num_timepoints):
            stacked = np.stack([d[t] for d in all_fov_data])
            expected_mean = float(np.nanmean(stacked))
            np.testing.assert_allclose(plate_tp_stats[str(t)]["mean"], expected_mean, rtol=1e-5)


def test_normalization_metadata_keys(small_hcs_dataset):
    """Each FOV must have fov_statistics, timepoint_statistics, and dataset_statistics."""
    generate_normalization_metadata(small_hcs_dataset, num_workers=1, grid_spacing=GRID_SPACING)

    with open_ome_zarr(small_hcs_dataset, mode="r") as plate:
        for channel in plate.channel_names:
            for _, fov in plate.positions():
                norm = fov.zattrs["normalization"][channel]
                assert "fov_statistics" in norm
                assert "timepoint_statistics" in norm
                assert "dataset_statistics" in norm


def test_compute_otsu_stores_threshold(bimodal_hcs_dataset):
    """compute_otsu=True stores otsu_threshold in fov_statistics."""
    generate_normalization_metadata(
        bimodal_hcs_dataset, num_workers=1, grid_spacing=GRID_SPACING, compute_otsu=True, otsu_grid_spacing=4
    )

    with open_ome_zarr(bimodal_hcs_dataset, mode="r") as plate:
        for _, fov in plate.positions():
            for channel in plate.channel_names:
                fov_stats = fov.zattrs["normalization"][channel]["fov_statistics"]
                assert "otsu_threshold" in fov_stats
                assert isinstance(fov_stats["otsu_threshold"], float)


def test_compute_otsu_threshold_separates_bimodal(bimodal_hcs_dataset):
    """Otsu threshold on bimodal fluorescence falls between the two modes."""
    generate_normalization_metadata(
        bimodal_hcs_dataset, num_workers=1, grid_spacing=GRID_SPACING, compute_otsu=True, otsu_grid_spacing=4
    )

    with open_ome_zarr(bimodal_hcs_dataset, mode="r") as plate:
        for _, fov in plate.positions():
            # Fluorescence has BG ~0.25 and FG ~5.25; threshold should separate them
            threshold = fov.zattrs["normalization"]["Fluorescence"]["fov_statistics"]["otsu_threshold"]
            assert 0.3 < threshold < 5.0, f"Otsu threshold {threshold} not between modes"


def test_compute_otsu_false_omits_threshold(small_hcs_dataset):
    """compute_otsu=False (default) does not store otsu_threshold."""
    generate_normalization_metadata(small_hcs_dataset, num_workers=1, grid_spacing=GRID_SPACING)

    with open_ome_zarr(small_hcs_dataset, mode="r") as plate:
        for _, fov in plate.positions():
            for channel in plate.channel_names:
                fov_stats = fov.zattrs["normalization"][channel]["fov_statistics"]
                assert "otsu_threshold" not in fov_stats


def test_generate_fg_masks_stores_mask(bimodal_hcs_dataset):
    """generate_fg_masks stores a uint8 binary mask array per position."""
    from viscy_utils.meta_utils import generate_fg_masks

    generate_normalization_metadata(
        bimodal_hcs_dataset, num_workers=1, grid_spacing=GRID_SPACING, compute_otsu=True, otsu_grid_spacing=4
    )
    generate_fg_masks(bimodal_hcs_dataset, channel_names=["Fluorescence"], num_workers=1)

    with open_ome_zarr(bimodal_hcs_dataset, mode="r") as plate:
        for _, fov in plate.positions():
            assert "fg_mask" in fov
            mask_arr = fov["fg_mask"]
            assert mask_arr.shape == fov["0"].shape
            mask_data = mask_arr[:]
            assert mask_data.dtype == np.uint8
            assert set(np.unique(mask_data)).issubset({0, 1})


def test_generate_fg_masks_separates_bimodal(bimodal_hcs_dataset):
    """Fluorescence mask marks the bright half as foreground."""
    from viscy_utils.meta_utils import generate_fg_masks

    generate_normalization_metadata(
        bimodal_hcs_dataset, num_workers=1, grid_spacing=GRID_SPACING, compute_otsu=True, otsu_grid_spacing=4
    )
    generate_fg_masks(bimodal_hcs_dataset, channel_names=["Fluorescence"], num_workers=1)

    with open_ome_zarr(bimodal_hcs_dataset, mode="r") as plate:
        ch_idx = plate.channel_names.index("Fluorescence")
        for _, fov in plate.positions():
            mask = fov["fg_mask"][:, ch_idx]
            assert mask.sum() > 0, "Mask should have foreground voxels"
            # Phase channel should be all zeros (not in channel_names)
            phase_idx = plate.channel_names.index("Phase")
            assert fov["fg_mask"][:, phase_idx].sum() == 0


def test_generate_fg_masks_requires_otsu(bimodal_hcs_dataset):
    """generate_fg_masks raises KeyError without prior Otsu computation."""
    from viscy_utils.meta_utils import generate_fg_masks

    with pytest.raises(KeyError):
        generate_fg_masks(bimodal_hcs_dataset, channel_names=["Fluorescence"], num_workers=1)


def test_generate_fg_masks_no_overwrite(bimodal_hcs_dataset):
    """generate_fg_masks raises FileExistsError on re-run."""
    from viscy_utils.meta_utils import generate_fg_masks

    generate_normalization_metadata(
        bimodal_hcs_dataset, num_workers=1, grid_spacing=GRID_SPACING, compute_otsu=True, otsu_grid_spacing=4
    )
    generate_fg_masks(bimodal_hcs_dataset, channel_names=["Fluorescence"], num_workers=1)

    with pytest.raises(FileExistsError):
        generate_fg_masks(bimodal_hcs_dataset, channel_names=["Fluorescence"], num_workers=1)
