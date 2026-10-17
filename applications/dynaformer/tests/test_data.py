"""Clip sampling, length bucketing and the batch contract."""

import numpy as np
import pytest
import torch

from viscy_data.sampler import LengthBucketedBatchSampler
from viscy_data.track_window import _contiguous_runs


def _fit_datasets(datamodule):
    """Set up the fit stage and return the train and validation datasets."""
    datamodule.setup("fit")
    return datamodule.train_dataset, datamodule.val_dataset


def test_contiguous_runs_splits_on_gaps():
    """A missing timepoint breaks a track into separate clips."""
    assert _contiguous_runs(np.array([0, 1, 2, 5, 6])) == [(0, 3), (3, 2)]
    assert _contiguous_runs(np.array([4])) == [(0, 1)]
    assert _contiguous_runs(np.array([], dtype=int)) == []


def test_short_tracks_contribute_one_full_length_clip(make_datamodule, track_spec):
    """Tracks shorter than the window are kept, not dropped."""
    datamodule = make_datamodule()
    train, val = _fit_datasets(datamodule)
    lengths = set(train.window_lengths().tolist()) | set(val.window_lengths().tolist())
    assert track_spec["short_length"] < datamodule.window_size
    assert track_spec["short_length"] in lengths
    assert datamodule.window_size in lengths


def test_clips_below_min_frames_are_dropped(make_datamodule, track_spec):
    """Raising min_frames above a track's length removes that track's clips."""
    short_length = track_spec["short_length"]
    train, val = _fit_datasets(make_datamodule(min_frames=short_length + 1))
    for dataset in (train, val):
        assert dataset.window_lengths().min() > short_length


def test_train_and_validation_tracks_are_disjoint(make_datamodule):
    """No track appears in both splits, so a clip never straddles the boundary."""
    train, val = _fit_datasets(make_datamodule(split_ratio=0.6))
    train_ids = set(train.tracks["global_track_id"])
    val_ids = set(val.tracks["global_track_id"])
    assert train_ids and val_ids
    assert not train_ids & val_ids


def test_batch_shapes_and_keys(make_datamodule, track_spec):
    """A batch carries registered source, target, mask and pooled scalar targets."""
    datamodule = make_datamodule()
    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))
    batch = datamodule.on_after_batch_transfer(batch, 0)

    clips, frames = batch["source"].shape[:2]
    patch = track_spec["final_patch"]
    n_target = len(track_spec["target_channels"])
    assert batch["source"].shape == (clips, frames, 1, 1, patch, patch)
    assert batch["target"].shape == (clips, frames, n_target, 1, patch, patch)
    assert batch["mask"].shape == (clips, frames, 1, 1, patch, patch)
    assert batch["scalar_target"].shape == (clips, frames, n_target)
    assert torch.isfinite(batch["scalar_target"]).all()
    assert batch["norm_meta"] is not None


def test_clip_frames_share_one_augmentation_draw(make_datamodule, track_spec):
    """One flip applies to every frame of a clip, keeping the clip registered.

    Flipping frames independently would break the spatial correspondence the
    temporal model relies on, and no shape assertion would catch it.
    """
    from viscy_transforms import BatchedRandFlipd

    channels = [track_spec["source_channel"], *track_spec["target_channels"]]
    augmented = make_datamodule(augmentations=[BatchedRandFlipd(keys=channels, spatial_axes=[1, 2], prob=1.0)])
    plain = make_datamodule()
    augmented.setup("fit")
    plain.setup("fit")
    run = int(np.argmax(plain.train_dataset.run_lengths()))

    flipped = augmented.on_after_batch_transfer(augmented.train_dataset.read_run(run), 0)
    unflipped = plain.on_after_batch_transfer(plain.train_dataset.read_run(run), 0)
    assert torch.allclose(flipped["source"], torch.flip(unflipped["source"], dims=(-2, -1)), atol=1e-5)


def test_border_tracks_are_clamped_not_dropped(make_datamodule, track_spec):
    """A track hugging the edge keeps every frame; its patch center is clamped.

    Dropping border frames instead would fragment long tracks into short runs,
    losing exactly the trajectories a temporal model needs.
    """
    import pandas as pd

    train, val = _fit_datasets(make_datamodule())
    border_id = track_spec["border_track_id"]
    border = pd.concat([dataset.tracks[dataset.tracks["track_id"] == border_id] for dataset in (train, val)])
    assert len(border) == track_spec["track_lengths"][border_id] * 2  # one copy per FOV
    assert (border["y_center"] < border["y"]).all()
    assert (border["x_center"] > border["x"]).all()


def test_read_run_returns_the_whole_track(make_datamodule, track_spec):
    """Whole-track inference gets every frame of a segment in one clip."""
    train, _ = _fit_datasets(make_datamodule())
    longest = int(np.argmax(train.run_lengths()))
    assert train.read_run(longest)["source"].shape[:2] == (1, track_spec["timepoints"])
    assert train.read_run(longest, max_frames=3)["source"].shape[1] == 3


def test_mismatched_clip_lengths_are_rejected(make_datamodule):
    """Reading clips of different lengths together fails loudly."""
    train, _ = _fit_datasets(make_datamodule())
    lengths = train.window_lengths()
    with pytest.raises(ValueError, match="clips of one length"):
        train.__getitems__([int(np.argmax(lengths)), int(np.argmin(lengths))])


def test_bucket_sampler_yields_one_length_per_batch():
    """Every batch holds clips of a single length, so no padding mask is needed."""
    lengths = np.array([8] * 10 + [3] * 7)
    sampler = LengthBucketedBatchSampler(lengths, frames_per_batch=16, shuffle=False)
    batches = list(sampler)
    assert len(batches) == len(sampler)
    for batch in batches:
        assert len({lengths[i] for i in batch}) == 1
    assert sorted(i for batch in batches for i in batch) == list(range(len(lengths)))


def test_bucket_sampler_keeps_the_frame_budget():
    """Shorter clips pack more per batch, keeping frames per batch roughly constant."""
    lengths = np.array([8] * 8 + [2] * 8)
    sampler = LengthBucketedBatchSampler(lengths, frames_per_batch=16, shuffle=False)
    assert sampler.batch_size_for(8) == 2
    assert sampler.batch_size_for(2) == 8
    for batch in sampler:
        assert len(batch) * lengths[batch[0]] <= 16


def test_bucket_sampler_splits_across_replicas():
    """Ranks receive disjoint batches and the same batch count."""
    lengths = np.array([4] * 24)
    batches = [
        list(LengthBucketedBatchSampler(lengths, frames_per_batch=8, shuffle=False, num_replicas=2, rank=rank))
        for rank in (0, 1)
    ]
    assert len(batches[0]) == len(batches[1])
    flat = [tuple(batch) for rank_batches in batches for batch in rank_batches]
    assert len(flat) == len(set(flat))


def test_bucket_sampler_reshuffles_each_epoch():
    """Successive iterations draw different batch orders."""
    sampler = LengthBucketedBatchSampler(np.array([4] * 32), frames_per_batch=8, seed=0)
    assert list(sampler) != list(sampler)
