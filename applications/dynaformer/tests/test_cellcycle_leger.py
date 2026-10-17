"""Cell-cycle regression on the Leger et al. RPE1 tracks: index, chunked embedding cache, clips, training, export."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from lightning.pytorch import Trainer
from torch import nn

from dynaformer.cellcycle_data import N_VIEWS, d4_view
from dynaformer.cellcycle_engine import TemporalEmbeddingModule
from dynaformer.leger_data import LegerDataModule, load_index, plan_chunks, track_table, write_index
from dynaformer.leger_embed import embed_chunk, finalize, plan
from dynaformer.leger_export import export, run_name
from dynaformer.models import MLPRegressionHead, RoPETransformerEncoder, TemporalEmbeddingModel
from viscy_utils.compose import load_composed_config

SIZE = 8
# (split, name, length): movie 0608 is the validation movie; tracks longer and shorter than the training window.
TRACKS = [
    ("train", "0403.12", 20),
    ("train", "0403.7", 9),
    ("train", "0503.1", 13),
    ("train", "0608.3", 11),
    ("train", "0608.40", 17),
    ("test", "0303.5", 15),
    ("test", "0303.6", 10),
    ("test_drugged", "0304.2", 12),
]
WINDOW = 16


class ToyEncoder(nn.Module):
    """A cheap stand-in for FoundationFrameEncoder with the attributes the cache records."""

    name, weights, img_size, center_patches, rgb_window = "toy", "toy", SIZE, 1, (-6.0, 4.0)
    token_dim, patch_dim, embed_dim = 4, 2, 6

    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.proj = nn.Linear(SIZE * SIZE, self.embed_dim)
        with torch.no_grad():
            for parameter in self.proj.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(1))


def labels_of(length: int, seed: int) -> np.ndarray:
    """Stored-layout labels (2, T, 1): channel 0 rises, channel 1 falls, so a transposition error shows."""
    t = np.linspace(0, 1, length)
    noise = np.random.default_rng(seed).uniform(0, 0.05, length)
    return np.stack([t * 0.9 + noise, 1 - t * 0.9 - noise])[:, :, None].astype(np.float16)


@pytest.fixture
def leger_dir(tmp_path):
    """Write a small dataset in the Leger et al. layout."""
    rng = np.random.default_rng(0)
    for seed, (split, name, length) in enumerate(TRACKS):
        for kind in ("images", "labels"):
            (tmp_path / split / kind).mkdir(parents=True, exist_ok=True)
        np.save(
            tmp_path / split / "images" / f"{name}.npy", rng.standard_normal((length, 2, SIZE, SIZE)).astype(np.float16)
        )
        np.save(tmp_path / split / "labels" / f"{name}.npy", labels_of(length, seed))
    return tmp_path


def build_cache(data_dir: Path, out: Path, frames_per_chunk: int, batch_size: int = 5) -> Path:
    """Index, plan, embed every chunk and finalize, with the toy encoder on CPU."""
    write_index(data_dir, out)
    planned = plan(out, "toy", frames_per_chunk, encoder=ToyEncoder())
    for chunk in range(len(planned["chunks"])):
        embed_chunk(
            data_dir, out, "toy", chunk, batch_size=batch_size, num_workers=0, device="cpu", encoder=ToyEncoder()
        )
    finalize(out, "toy")
    return out / "toy.npy"


def test_index_holds_every_frame_with_transposed_labels(leger_dir):
    index_dir = leger_dir / "embeddings"
    write_index(leger_dir, index_dir)
    index, labels = load_index(index_dir)
    assert len(index) == sum(length for _, _, length in TRACKS) == len(labels)
    assert index.split.tolist() == sorted(index.split, key=["train", "test", "test_drugged"].index)
    assert index.movie.iloc[0] == "0403" and {"0403.7", "0403.12"} <= set(index["name"])  # strings, not numbers
    for _, track in index.groupby("track_index"):
        name, split = track["name"].iloc[0], track.split.iloc[0]
        stored = np.load(leger_dir / split / "labels" / f"{name}.npy")
        assert (np.diff(track.row) == 1).all() and (track.t.to_numpy() == np.arange(len(track))).all()
        assert np.array_equal(labels[track.row], stored[:, :, 0].T.astype(np.float32))
    assert labels[index.row[index["name"] == "0403.12"], 0][-1] > 0.8  # Geminin (channel 0) rises along the track


def test_chunks_are_whole_tracks_covering_every_row(leger_dir):
    write_index(leger_dir, leger_dir / "embeddings")
    tracks = track_table(load_index(leger_dir / "embeddings")[0])
    for frames_per_chunk in (1, 25, 40, 1000):
        chunks = plan_chunks(tracks, frames_per_chunk)
        assert chunks[0][0] == 0 and chunks[-1][1] == tracks.length.sum()
        assert all(a[1] == b[0] for a, b in zip(chunks, chunks[1:], strict=False))
        starts = set(tracks.row)
        assert all(start in starts for start, _ in chunks)
        assert all(end - start <= frames_per_chunk or (end - start) in set(tracks.length) for start, end in chunks)


def test_chunked_cache_is_the_frames_embedding_in_every_view(leger_dir, tmp_path):
    """Rows line up with the index whatever the chunking, and finalize refuses an unwritten chunk."""
    chunked = np.load(build_cache(leger_dir, tmp_path / "chunked", frames_per_chunk=25))
    whole = np.load(build_cache(leger_dir, tmp_path / "whole", frames_per_chunk=10_000, batch_size=64))
    assert len(json.loads((tmp_path / "chunked" / "toy.json").read_text())["chunks"]) > 2
    np.testing.assert_allclose(chunked.astype(np.float32), whole.astype(np.float32), rtol=2e-3, atol=2e-3)

    index, _ = load_index(tmp_path / "chunked")
    encoder = ToyEncoder()
    for row in (0, 19, 20, int(index.row.iloc[-1])):
        name, split, t = index.loc[row, ["name", "split", "t"]]
        frame = torch.from_numpy(np.load(leger_dir / split / "images" / f"{name}.npy")[t, :1].astype(np.float32))
        for view in range(N_VIEWS):
            expected = encoder(d4_view(frame[None], view)).detach().numpy()[0]
            np.testing.assert_allclose(chunked[row, view].astype(np.float32), expected, rtol=2e-3, atol=2e-3)

    (tmp_path / "chunked" / "toy.chunks" / "1.json").unlink()
    with pytest.raises(RuntimeError, match="Chunk 1"):
        finalize(tmp_path / "chunked", "toy")


def test_drawn_weights_travel_with_the_plan(leger_dir, tmp_path, monkeypatch):
    """A chunk that draws other random weights still embeds with the planned draw; without a saved draw it refuses."""
    other_draw = ToyEncoder()
    with torch.no_grad():
        other_draw.proj.weight.add_(1e-3)
    monkeypatch.setattr("dynaformer.leger_embed.DRAWN_WEIGHTS", ("toy",))
    write_index(leger_dir, tmp_path / "saved")
    plan(tmp_path / "saved", "toy", 30, encoder=ToyEncoder())
    embed_chunk(leger_dir, tmp_path / "saved", "toy", 0, num_workers=0, device="cpu", encoder=other_draw)
    index, _ = load_index(tmp_path / "saved")
    frame = torch.from_numpy(np.load(leger_dir / "train" / "images" / f"{index['name'][0]}.npy")[0, :1].astype("f4"))
    expected = ToyEncoder()(frame[None]).detach().numpy()[0]
    cached = np.load(tmp_path / "saved" / "toy.npy")[0, 0].astype(np.float32)
    np.testing.assert_allclose(cached, expected, rtol=2e-3, atol=2e-3)

    monkeypatch.setattr("dynaformer.leger_embed.DRAWN_WEIGHTS", ())
    write_index(leger_dir, tmp_path / "unsaved")
    plan(tmp_path / "unsaved", "toy", 30, encoder=ToyEncoder())
    perturbed = ToyEncoder()
    with torch.no_grad():
        perturbed.proj.bias.add_(1e-3)
    with pytest.raises(ValueError, match="differ"):
        embed_chunk(leger_dir, tmp_path / "unsaved", "toy", 0, num_workers=0, device="cpu", encoder=perturbed)


def make_datamodule(cache: Path, **overrides) -> LegerDataModule:
    kwargs = {
        "embeddings": str(cache),
        "window_size": WINDOW,
        "length_quantum": 4,
        "min_frames": 4,
        "frames_per_batch": 40,
        "eval_frames_per_batch": 40,
        "num_workers": 0,
    }
    kwargs.update(overrides)
    return LegerDataModule(**kwargs)


def test_batches_read_the_cache_rows_and_their_labels(leger_dir, tmp_path):
    cache_path = build_cache(leger_dir, tmp_path / "cache", frames_per_chunk=30)
    cache = np.load(cache_path)
    datamodule = make_datamodule(cache_path)
    datamodule.setup("fit")
    index, labels = datamodule.index, datamodule.labels
    train_names = {index["name"][index.track_index == clip[3]].iloc[0] for clip in datamodule.train_dataset.clips}
    val_names = {index["name"][index.track_index == clip[3]].iloc[0] for clip in datamodule.val_dataset.clips}
    assert train_names == {"0403.12", "0403.7", "0503.1"} and val_names == {"0608.3", "0608.40"}

    seen = set()
    for batch in datamodule.train_dataloader():
        clips, frames = batch["rows"].shape
        assert frames <= WINDOW and batch["embeddings"].shape == (clips, frames, ToyEncoder.embed_dim)
        for i in range(clips):
            rows, view = batch["rows"][i].numpy(), int(batch["view"][i])
            track = index.loc[rows]
            assert (track.track_index == int(batch["track_id"][i])).all()
            assert np.array_equal(batch["t"][i].numpy(), track.t.to_numpy())
            assert np.array_equal(batch["embeddings"][i].numpy(), cache[rows, view].astype(np.float32))
            assert np.array_equal(batch["target"][i].numpy(), labels[rows])
            seen.add(int(batch["track_id"][i]))
    assert len(seen) == 3  # one clip per training track and epoch

    for batch in datamodule.val_dataloader():
        assert (batch["view"] == 0).all()
        for i, rows in enumerate(batch["rows"].numpy()):
            track = index[index.track_index == int(batch["track_id"][i])]
            assert np.array_equal(rows, track.row.to_numpy())  # whole tracks

    datamodule = make_datamodule(cache_path, test_split="test_drugged")
    datamodule.setup("test")
    assert [c[1] for c in datamodule.test_dataset.clips] == [12]


def test_a_cache_needs_its_finalize_and_its_index(leger_dir, tmp_path):
    cache_path = build_cache(leger_dir, tmp_path / "cache", frames_per_chunk=30)
    index = pd.read_csv(tmp_path / "cache" / "index.csv", dtype=str)
    index.loc[0, "t"] = "1"
    index.to_csv(tmp_path / "cache" / "index.csv", index=False)
    with pytest.raises(ValueError, match="different"):
        make_datamodule(cache_path)
    cache_path.with_suffix(".json").unlink()
    with pytest.raises(FileNotFoundError, match="not finalized"):
        make_datamodule(cache_path)


def test_fit_test_and_export_round_trip(leger_dir, tmp_path):
    """Train, score each split, and export test predictions in the harness format with the stored labels."""
    cache_path = build_cache(leger_dir, tmp_path / "cache", frames_per_chunk=30)
    model = TemporalEmbeddingModel(
        ToyEncoder.embed_dim,
        temporal=RoPETransformerEncoder(d_model=16, n_heads=2, n_layers=1),
        regression_head=MLPRegressionHead(16, hidden_dims=8),
    )
    run_dir = tmp_path / "run"
    module = TemporalEmbeddingModule(model, warmup_steps=1, output_dir=str(run_dir / "eval" / "best_regression"))
    trainer = Trainer(
        max_epochs=2,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(module, datamodule=make_datamodule(cache_path))
    assert "val/regression/track_pearson_cdt1" in trainer.callback_metrics
    assert not any("phase" in key for key in trainer.callback_metrics)  # no PIP-FUCCI phase rule here
    for split in ("test", "test_drugged"):
        trainer.test(module, datamodule=make_datamodule(cache_path, test_split=split))
    predictions = pd.read_csv(run_dir / "eval" / "best_regression" / "predictions_test.csv")
    assert len(predictions) == 25 and {"true_geminin", "regression_cdt1"} <= set(predictions.columns)

    out = tmp_path / "harness"
    path = export(run_dir, run_name("regression_toy.yml"), "test", leger_dir, tmp_path / "cache", out)
    assert path.name == "dynaformer_toy_temporal_test.npz"
    data = np.load(path)
    assert data["names"].tolist() == ["0303.5", "0303.6"]
    for name in data["names"]:
        stored = np.load(leger_dir / "test" / "labels" / f"{name}.npy")[:, :, 0].T.astype(np.float32)
        assert np.array_equal(data[f"true_{name}"], stored)
        assert data[f"pred_{name}"].shape == stored.shape
    track_index = load_index(tmp_path / "cache")[0].query("name == '0303.6'").track_index.iloc[0]
    track = predictions[predictions.track_id == track_index].sort_values("t")
    np.testing.assert_array_equal(data["pred_0303.6"], track[["regression_geminin", "regression_cdt1"]].to_numpy("f4"))
    assert export(run_dir, "x", "test_drugged", leger_dir, tmp_path / "cache", out).exists()

    predictions.iloc[:-1].to_csv(run_dir / "eval" / "best_regression" / "predictions_test.csv", index=False)
    with pytest.raises(ValueError, match="cover"):
        export(run_dir, "x", "test", leger_dir, tmp_path / "cache", out)


def test_run_names():
    assert run_name("regression_c-radio.yml") == "dynaformer_c-radio_temporal"
    assert run_name("regression_vit-untrained_single_frame.yml") == "dynaformer_vit-untrained_single_frame"
    with pytest.raises(ValueError):
        run_name("stain_c-radio.yml")


CONFIGS = Path(__file__).resolve().parents[1] / "configs"
EMBED_DIMS = {"morphem": 768, "cell-dino": 2048, "dinov3": 2048, "c-radio": 3840, "vit-untrained": 768}


@pytest.mark.parametrize(
    "leaf", sorted(path.name for path in (CONFIGS / "cellcycle_leger" / "experiments").glob("*.yml"))
)
def test_leger_leaves_train_the_nih3t3_regression_model(leaf):
    """Same model, optimizer, budget and checkpointing as the NIH3T3 leaf of the same name; only the data differ."""
    config = load_composed_config(CONFIGS / "cellcycle_leger" / "experiments" / leaf)
    reference = load_composed_config(CONFIGS / "cellcycle" / "experiments" / leaf)
    assert config["model"] == reference["model"]
    data = config["data"]
    assert data["class_path"] == "dynaformer.leger_data.LegerDataModule"
    assert (
        config["model"]["init_args"]["model"]["init_args"]["embed_dim"]
        == EMBED_DIMS[Path(data["init_args"]["embeddings"]).stem]
    )
    trainer, reference_trainer = dict(config["trainer"]), dict(reference["trainer"])
    assert trainer.pop("logger")["init_args"]["project"] == "dynaformer-cellcycle-leger"
    reference_trainer.pop("logger")
    assert trainer == reference_trainer
