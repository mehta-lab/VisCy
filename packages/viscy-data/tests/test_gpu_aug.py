"""Tests for CachedOmeZarrDataset and CachedOmeZarrDataModule."""

import torch
from iohub import open_ome_zarr
from torch.multiprocessing import Manager

from viscy_data import CachedOmeZarrDataset


def test_cached_dataset_length(preprocessed_hcs_dataset):
    """Dataset length equals total timepoints across all positions."""
    with open_ome_zarr(preprocessed_hcs_dataset) as plate:
        positions = [p for _, p in plate.positions()]
        channel_names = plate.channel_names
        expected_len = sum(p["0"].frames for p in positions)

    cache_map = Manager().dict()
    dataset = CachedOmeZarrDataset(
        positions=positions,
        channel_names=channel_names[:2],
        cache_map=cache_map,
    )
    assert len(dataset) == expected_len


def test_cached_dataset_returns_correct_channels(preprocessed_hcs_dataset):
    """Dataset returns one key per requested channel."""
    with open_ome_zarr(preprocessed_hcs_dataset) as plate:
        positions = [p for _, p in plate.positions()]
        channel_names = plate.channel_names[:2]

    cache_map = Manager().dict()
    dataset = CachedOmeZarrDataset(
        positions=positions,
        channel_names=channel_names,
        cache_map=cache_map,
    )
    sample = dataset[0]
    # Returns a list of samples (for MultiSampleTrait compat)
    assert isinstance(sample, list)
    sample = sample[0]
    for ch in channel_names:
        assert ch in sample
        assert isinstance(sample[ch], torch.Tensor)


def test_cached_dataset_caching(preprocessed_hcs_dataset):
    """Second access should use cached value."""
    with open_ome_zarr(preprocessed_hcs_dataset) as plate:
        positions = [p for _, p in plate.positions()]
        channel_names = plate.channel_names[:1]

    cache_map = Manager().dict()
    dataset = CachedOmeZarrDataset(
        positions=positions,
        channel_names=channel_names,
        cache_map=cache_map,
    )
    # First access loads from zarr
    assert cache_map[0] is None
    _ = dataset[0]
    # Second access should use cache
    assert cache_map[0] is not None
    sample_cached = dataset[0]
    assert isinstance(sample_cached, list)


def test_cached_dataset_skip_cache(preprocessed_hcs_dataset):
    """With skip_cache=True, cache_map stays None after access."""
    with open_ome_zarr(preprocessed_hcs_dataset) as plate:
        positions = [p for _, p in plate.positions()]
        channel_names = plate.channel_names[:1]

    cache_map = Manager().dict()
    dataset = CachedOmeZarrDataset(
        positions=positions,
        channel_names=channel_names,
        cache_map=cache_map,
        skip_cache=True,
    )
    _ = dataset[0]
    assert cache_map[0] is None


def test_cached_dataset_norm_meta(preprocessed_hcs_dataset):
    """Dataset loads normalization metadata when requested."""
    with open_ome_zarr(preprocessed_hcs_dataset) as plate:
        positions = [p for _, p in plate.positions()]
        channel_names = plate.channel_names[:2]

    cache_map = Manager().dict()
    dataset = CachedOmeZarrDataset(
        positions=positions,
        channel_names=channel_names,
        cache_map=cache_map,
        load_normalization_metadata=True,
    )
    sample = dataset[0][0]
    assert "norm_meta" in sample

    # Without norm_meta
    cache_map2 = Manager().dict()
    dataset_no_norm = CachedOmeZarrDataset(
        positions=positions,
        channel_names=channel_names,
        cache_map=cache_map2,
        load_normalization_metadata=False,
    )
    sample_no_norm = dataset_no_norm[0][0]
    assert "norm_meta" not in sample_no_norm
