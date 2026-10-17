import numpy as np
import pytest
import torch
from viscy_data.npy_track import NpyTrackDataModule, NpyTrackDataset

MOVIES = {"0601": 4, "0602": 3, "0608": 3}
TRACK_LENGTHS = [40, 200, 96, 12]
PATCH = 64
N_IMAGE_CHANNELS = 2
N_TARGET_CHANNELS = 2


@pytest.fixture
def npy_root(tmp_path):
    """Write a dataset in the released layout: (T, C, Y, X) images and (C, T, 1) labels."""
    rng = np.random.default_rng(0)
    for split in ("train", "test"):
        root = tmp_path / split
        (root / "images").mkdir(parents=True)
        (root / "labels").mkdir(parents=True)
        for movie, n_tracks in MOVIES.items():
            for track in range(n_tracks):
                frames = TRACK_LENGTHS[track % len(TRACK_LENGTHS)]
                images = rng.standard_normal((frames, N_IMAGE_CHANNELS, PATCH, PATCH), dtype=np.float32)
                labels = rng.random((N_TARGET_CHANNELS, frames, 1), dtype=np.float32)
                np.save(root / "images" / f"{movie}.{track}.npy", images.astype(np.float16))
                np.save(root / "labels" / f"{movie}.{track}.npy", labels.astype(np.float16))
    return tmp_path


def make_datamodule(npy_root, **overrides):
    kwargs = dict(
        data_path=str(npy_root),
        window_lengths=(4, 16, 32),
        frames_per_batch=64,
        num_workers=0,
        val_movies=("0608",),
    )
    kwargs.update(overrides)
    return NpyTrackDataModule(**kwargs)


def test_batch_shapes_match_the_model_contract(npy_root):
    datamodule = make_datamodule(npy_root)
    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))

    clips, frames = batch["source"].shape[:2]
    # The model requires a 6-D (B, T, C, Z, Y, X) movie; the singleton Z is not optional.
    assert batch["source"].shape == (clips, frames, 1, 1, PATCH, PATCH)
    assert batch["scalar_target"].shape == (clips, frames, N_TARGET_CHANNELS)
    assert torch.isfinite(batch["source"]).all()
    assert torch.isfinite(batch["scalar_target"]).all()


def test_every_batch_has_a_single_clip_length(npy_root):
    datamodule = make_datamodule(npy_root)
    datamodule.setup("fit")
    for batch in datamodule.train_dataloader():
        frames = batch["source"].shape[1]
        assert frames in (4, 16, 32)
        assert batch["scalar_target"].shape[1] == frames


def test_frame_budget_caps_the_batch(npy_root):
    datamodule = make_datamodule(npy_root, frames_per_batch=64)
    datamodule.setup("fit")
    for batch in datamodule.train_dataloader():
        clips, frames = batch["source"].shape[:2]
        assert clips * frames <= 64


def test_split_is_by_movie_not_by_track(npy_root):
    datamodule = make_datamodule(npy_root)
    datamodule.setup("fit")
    train_movies = {n.split(".")[0] for n in datamodule.train_dataset.names}
    val_movies = {n.split(".")[0] for n in datamodule.val_dataset.names}
    assert val_movies == {"0608"}
    assert not train_movies & val_movies


def test_unknown_validation_movie_raises(npy_root):
    datamodule = make_datamodule(npy_root, val_movies=("9999",))
    with pytest.raises(ValueError, match="No tracks from movies"):
        datamodule.setup("fit")


def test_whole_track_mode_returns_full_length_clips(npy_root):
    datamodule = make_datamodule(npy_root)
    datamodule.setup("fit")
    lengths = []
    for batch in datamodule.val_dataloader():
        assert batch["source"].shape[0] == 1
        lengths.append(batch["source"].shape[1])
    assert sorted(lengths) == sorted(datamodule.val_dataset.track_lengths)


def test_offsets_change_between_epochs_but_lengths_do_not(npy_root):
    dataset = NpyTrackDataset(npy_root / "train", ["0601.1"], source_channel=0, window_lengths=(16,), seed=7)
    dataset.set_epoch(0)
    first = dataset[0]["source"].clone()
    dataset.set_epoch(1)
    second = dataset[0]["source"]
    assert first.shape == second.shape
    assert not torch.equal(first, second), "clip offsets should be redrawn each epoch"
    assert dataset.window_lengths().tolist() == [16]


def test_source_channel_selects_the_requested_image_channel(npy_root):
    raw = np.load(npy_root / "train" / "images" / "0601.1.npy").astype(np.float32)
    for channel in (0, 1):
        dataset = NpyTrackDataset(npy_root / "train", ["0601.1"], source_channel=channel, whole_track=True)
        clip = dataset[0]["source"][0, :, 0, 0]
        assert torch.allclose(clip, torch.from_numpy(raw[:, channel]))


def test_labels_are_transposed_out_of_the_stored_layout(npy_root):
    raw = np.load(npy_root / "train" / "labels" / "0601.1.npy").astype(np.float32)
    dataset = NpyTrackDataset(npy_root / "train", ["0601.1"], whole_track=True)
    target = dataset[0]["scalar_target"][0]
    assert target.shape == (raw.shape[1], N_TARGET_CHANNELS)
    assert torch.allclose(target, torch.from_numpy(raw[:, :, 0].T))


def test_tracks_shorter_than_every_window_length_are_dropped(npy_root):
    # The shortest fixture track is 12 frames, so a 32-frame minimum excludes it.
    dataset = NpyTrackDataset(npy_root / "train", _all_names(npy_root), source_channel=0, window_lengths=(32,))
    assert dataset.window_lengths().tolist() == [32] * sum(1 for t in _lengths(npy_root) if t >= 32)


def test_image_target_is_emitted_only_when_requested(npy_root):
    without = NpyTrackDataset(npy_root / "train", ["0601.1"], whole_track=True)
    assert "target" not in without[0]

    with_target = NpyTrackDataset(npy_root / "train", ["0601.1"], target_channel=1, whole_track=True)
    batch = with_target[0]
    assert batch["target"].shape == batch["source"].shape
    raw = np.load(npy_root / "train" / "images" / "0601.1.npy").astype(np.float32)
    assert torch.allclose(batch["target"][0, :, 0, 0], torch.from_numpy(raw[:, 1]))
    # source must still be the source channel, not a copy of the target
    assert not torch.equal(batch["source"], batch["target"])


def test_datamodule_threads_the_image_target_through(npy_root):
    datamodule = make_datamodule(npy_root, target_channel=1)
    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))
    assert batch["target"].shape == batch["source"].shape
    assert torch.isfinite(batch["target"]).all()


def test_no_fitting_window_raises(npy_root):
    with pytest.raises(ValueError, match="No clips of any length"):
        NpyTrackDataset(npy_root / "train", _all_names(npy_root), window_lengths=(10_000,))


def _all_names(npy_root):
    return sorted(p.stem for p in (npy_root / "train" / "images").glob("*.npy"))


def _lengths(npy_root):
    return [len(np.load(npy_root / "train" / "images" / f"{n}.npy", mmap_mode="r")) for n in _all_names(npy_root)]
