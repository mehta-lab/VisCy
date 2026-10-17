"""Cell-cycle prediction from cached foundation-model embeddings: data, model, training and scoring."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from lightning.pytorch import Trainer

from dynaformer.cellcycle_data import N_VIEWS, CellCycleDataModule, CellCycleDataset, d4_view, decode_phase
from dynaformer.cellcycle_engine import TemporalEmbeddingModule
from dynaformer.cellcycle_metrics import masked_median, phase_forced, phase_strict, score_frames
from dynaformer.embed import _CentredPhase
from dynaformer.models import ConditionedUNet, MLPRegressionHead, RoPETransformerEncoder, TemporalEmbeddingModel
from viscy_data.sampler import LengthBucketedBatchSampler
from viscy_utils.compose import load_composed_config

CROP = 96
SIZE = 64
EMBED = 24  # 16-wide token + 8-wide centre mean
TOKEN = 16
D_MODEL = 32
THRESHOLDS = {
    "gem_valley": 100.0,
    "pip_valley": 70.0,
    "gem_off": 60.0,
    "gem_on": 150.0,
    "pip_off": 42.0,
    "pip_on": 105.0,
}
# (split, length) of every track: some longer than the training window, some shorter, one below min_frames.
TRACKS = [("train", 12), ("train", 5), ("train", 9), ("train", 12), ("train", 2), ("val", 12), ("val", 6), ("test", 7)]


@pytest.fixture
def cellcycle_dir(tmp_path):
    """Write a small dataset folder in the layout of Bill/5-cellcycle, with an embedding cache."""
    rng = np.random.default_rng(0)
    rows, splits = [], []
    for track_id, (split, length) in enumerate(TRACKS, start=1):
        splits.append({"track_id": track_id, "split": split})
        for t in range(length):
            rows.append(
                {
                    "track_id": track_id,
                    "t": t,
                    "mko": float(rng.uniform(0, 1500)),
                    "mvenus": float(rng.uniform(0, 2000)),
                    "bg_mko": 50.0,
                    "bg_mvenus": 0.0,
                }
            )
    points = pd.DataFrame(rows)
    points.insert(0, "row", np.arange(len(points)))
    points.to_csv(tmp_path / "points.csv", index=False)
    pd.DataFrame(splits).to_csv(tmp_path / "split_tracks.csv", index=False)
    stats = {
        "crop": CROP,
        "phase": {"scale": 8000.0, "offset_neg": 3.0, "padding_value": 0, "mean": 0.0, "std": 0.2},
        "fucci": {"scale": [1500.0, 2000.0]},
        "thresholds": THRESHOLDS,
    }
    (tmp_path / "stats.json").write_text(json.dumps(stats))
    crops = tmp_path / "crops"
    crops.mkdir()
    n = len(points)
    np.save(crops / "phase.npy", rng.integers(20000, 28000, size=(n, CROP, CROP), dtype=np.uint16))
    np.save(crops / "fucci.npy", rng.integers(0, 2000, size=(n, 2, CROP, CROP), dtype=np.uint16))
    yy, xx = np.mgrid[:CROP, :CROP]
    disk = ((yy - CROP / 2) ** 2 + (xx - CROP / 2) ** 2 < 15**2).astype(np.uint8)
    np.save(crops / "mask.npy", np.broadcast_to(disk, (n, CROP, CROP)).copy())
    embeddings = tmp_path / "embeddings"
    embeddings.mkdir()
    np.save(embeddings / "toy.npy", rng.standard_normal((n, N_VIEWS, EMBED)).astype(np.float16))
    meta = {"token_dim": TOKEN, "patch_dim": EMBED - TOKEN, "embed_dim": EMBED, "views": N_VIEWS}
    (embeddings / "toy.json").write_text(json.dumps(meta))
    return tmp_path


def make_datamodule(data_dir, images: bool, embeddings: bool = True, **overrides) -> CellCycleDataModule:
    kwargs = {
        "data_dir": str(data_dir),
        "embeddings": str(data_dir / "embeddings" / "toy.npy") if embeddings else None,
        "window_size": 8,
        "length_quantum": 4,
        "min_frames": 4,
        "frames_per_batch": 16,
        "eval_frames_per_batch": 16,
        "images_per_batch": 6 if images else 0,
        "val_image_stride": 3,
        "image_size": SIZE,
        "jitter": 8,
        "num_workers": 0,
    }
    kwargs.update(overrides)
    return CellCycleDataModule(**kwargs)


def make_model(task: str, n_layers: int = 1, cond_sigma: float | None = 16.0) -> TemporalEmbeddingModel:
    temporal = RoPETransformerEncoder(d_model=D_MODEL, n_heads=4, n_layers=n_layers)
    regression = MLPRegressionHead(D_MODEL, hidden_dims=16) if task in ("regression", "multitask") else None
    stain = None
    if task in ("stain", "multitask"):
        stain = ConditionedUNet(cond_dim=D_MODEL, cond_sigma=cond_sigma, backbone="convnextv2_atto")
    if task == "unconditioned":
        return TemporalEmbeddingModel(EMBED, stain_head=ConditionedUNet(backbone="convnextv2_atto"))
    return TemporalEmbeddingModel(EMBED, temporal=temporal, regression_head=regression, stain_head=stain)


def test_d4_views_match_between_numpy_and_torch():
    """The embedding cache (torch, on GPU) and the loader (NumPy) must rotate images identically."""
    image = np.arange(4 * 5 * 5, dtype=np.float32).reshape(4, 5, 5)
    seen = set()
    for view in range(N_VIEWS):
        expected = d4_view(image, view)
        assert np.array_equal(d4_view(torch.from_numpy(image), view).numpy(), expected)
        seen.add(expected.tobytes())
    assert len(seen) == N_VIEWS


def test_decode_phase_zeroes_padding():
    raw = np.array([[0, 24000], [32000, 16000]], dtype=np.uint16)
    z = decode_phase(raw, {"scale": 8000.0, "offset_neg": 3.0, "padding_value": 0, "mean": 0.0, "std": 0.5})
    assert z[0, 0] == 0
    assert np.allclose(z[0, 1:], [0.0]) and np.allclose(z[1], [2.0, -2.0])


def test_training_batches_pair_views_and_images(cellcycle_dir):
    """A clip's embeddings and images come from one view; only some frames get images."""
    datamodule = make_datamodule(cellcycle_dir, images=True, jitter=0)
    datamodule.setup("fit")
    dataset = datamodule.train_dataset
    loader = datamodule.train_dataloader()
    cache = np.load(cellcycle_dir / "embeddings" / "toy.npy")
    phase = np.load(cellcycle_dir / "crops" / "phase.npy")
    margin = (CROP - SIZE) // 2
    for batch in loader:
        clips, frames = batch["rows"].shape
        assert len(set(dataset.window_lengths().tolist())) > 1
        assert batch["embeddings"].shape == (clips, frames, EMBED)
        assert batch["target"].shape == (clips, frames, 2)
        assert len(batch["image_index"]) == min(6, clips * frames)
        assert batch["source"].shape == (len(batch["image_index"]), 1, SIZE, SIZE)
        assert batch["target_image"].shape == (len(batch["image_index"]), 2, SIZE, SIZE)
        rows, views = batch["rows"].numpy(), batch["view"].numpy()
        for clip in range(clips):
            expected = cache[rows[clip], views[clip]].astype(np.float32)
            assert np.allclose(batch["embeddings"][clip].numpy(), expected)
        for k, flat in enumerate(batch["image_index"].numpy()):
            clip, frame = divmod(int(flat), frames)
            window = phase[rows[clip, frame], margin : margin + SIZE, margin : margin + SIZE]
            expected = d4_view(decode_phase(window, datamodule.stats["phase"]), int(views[clip]))
            assert np.allclose(batch["source"][k, 0].numpy(), expected, atol=1e-5)


def test_phase_frames_are_what_the_cache_encodes(cellcycle_dir):
    """A trained encoder sees exactly the crops the cache encodes, in its clip's view, the same view as its images."""
    datamodule = make_datamodule(cellcycle_dir, images=True, embeddings=False, phase_frames=True, jitter=0)
    datamodule.setup("fit")
    centred = _CentredPhase(cellcycle_dir / "crops" / "phase.npy", datamodule.stats, SIZE)
    for batch in datamodule.train_dataloader():
        clips, frames = batch["rows"].shape
        assert "embeddings" not in batch
        assert batch["frames"].shape == (clips, frames, 1, SIZE, SIZE)
        rows, views = batch["rows"].numpy(), batch["view"].numpy()
        for clip in range(clips):
            for frame in range(frames):
                expected = d4_view(centred[rows[clip, frame]], int(views[clip]))
                assert torch.equal(batch["frames"][clip, frame], expected)
        for k, flat in enumerate(batch["image_index"].numpy()):
            assert torch.equal(batch["source"][k], batch["frames"].flatten(0, 1)[flat])
    with pytest.raises(ValueError, match="not both"):
        make_datamodule(cellcycle_dir, images=False, phase_frames=True).setup("fit")


def test_every_training_track_gives_one_clip_and_every_clip_is_batched():
    """Tracks, not windows, are the samples, and length bucketing must not leave any of them out of an epoch.

    With one bucket per exact length, most lengths hold fewer tracks than a batch and were silently dropped; this
    pins the quantized lengths and partial batches that prevent it, at the real dataset's scale.
    """
    rng = np.random.default_rng(0)
    lengths = rng.integers(8, 88, size=2722)
    points = pd.DataFrame(
        {
            "track_id": np.repeat(np.arange(len(lengths)), lengths),
            "t": np.concatenate([np.arange(n) for n in lengths]),
        }
    )
    points["row"] = np.arange(len(points))
    points[["mko", "mvenus", "bg_mko", "bg_mvenus"]] = 0.0
    stats = {"crop": CROP, "fucci": {"scale": [1.0, 1.0]}}
    dataset = CellCycleDataset(
        ".", points, stats, embeddings=None, window_size=64, length_quantum=8, min_frames=8, image_size=SIZE, jitter=8
    )
    assert len(dataset) == len(lengths)
    assert set(dataset.window_lengths().tolist()) == {8, 16, 24, 32, 40, 48, 56, 64}
    sampler = LengthBucketedBatchSampler(dataset.window_lengths(), frames_per_batch=2048, shuffle=True)
    assert sorted(i for batch in sampler for i in batch) == list(range(len(dataset)))


def test_validation_runs_whole_tracks_and_splits_are_disjoint(cellcycle_dir):
    datamodule = make_datamodule(cellcycle_dir, images=True)
    datamodule.setup("fit")
    val = datamodule.val_dataset
    assert sorted(val.window_lengths().tolist()) == [6, 12]
    train_tracks = set(datamodule.train_dataset.points.track_id)
    assert not train_tracks & set(val.points.track_id)
    # The 2-frame training track is below min_frames and gives no clip.
    assert all(length >= 4 for length in datamodule.train_dataset.window_lengths())
    batch = val.__getitems__([0])
    frames = batch["rows"].shape[1]
    assert batch["image_index"].tolist() == list(range(0, frames, 3))
    assert (batch["view"] == 0).all()


def test_film_is_identity_at_initialization():
    """Zero-initialized FiLM: a fresh conditioned UNet ignores its conditioning entirely."""
    torch.manual_seed(0)
    unet = ConditionedUNet(cond_dim=D_MODEL, backbone="convnextv2_atto").eval()
    images = torch.randn(3, 1, SIZE, SIZE)
    with torch.no_grad():
        a = unet(images, torch.randn(3, D_MODEL))
        b = unet(images, torch.randn(3, D_MODEL))
        film, unet.film = unet.film, None
        plain = unet(images)
        unet.film = film
    assert torch.equal(a, b)
    assert torch.equal(a, plain)


def test_centre_gate_confines_conditioning_to_the_tracked_cell():
    """With a narrow gate, changing the embedding changes the centre of the prediction far more than the corners."""
    torch.manual_seed(0)
    unet = ConditionedUNet(cond_dim=D_MODEL, cond_sigma=8.0, backbone="convnextv2_atto", decoder_conv_blocks=1).eval()
    for film in unet.film:
        torch.nn.init.normal_(film.proj.weight, std=0.5)
    images = torch.randn(1, 1, SIZE, SIZE)
    with torch.no_grad():
        change = (unet(images, torch.randn(1, D_MODEL)) - unet(images, torch.randn(1, D_MODEL))).abs()
    centre = change[..., 24:40, 24:40].mean()
    corners = torch.stack([change[..., :12, :12].mean(), change[..., -12:, -12:].mean()]).mean()
    assert centre > 3 * corners


def test_single_frame_ablation_has_no_temporal_mixing():
    """With n_layers=0 a frame's prediction depends on its own embedding only; with layers, on the others too."""
    torch.manual_seed(0)
    embeddings = torch.randn(1, 6, EMBED)
    perturbed = embeddings.clone()
    perturbed[0, 5] += 3.0 * torch.randn(EMBED)  # not a constant shift, which the input LayerNorm would remove
    for n_layers, mixes in ((0, False), (2, True)):
        model = make_model("regression", n_layers=n_layers).eval()
        with torch.no_grad():
            change = (model(perturbed)["regression"] - model(embeddings)["regression"])[0, :5].abs().max()
        assert (change > 1e-5) == mixes


def test_model_shapes_and_contracts():
    model = make_model("multitask").eval()
    embeddings = torch.randn(2, 5, EMBED)
    images = torch.randn(3, 1, SIZE, SIZE)
    out = model(embeddings, images, torch.tensor([0, 4, 9]))
    assert out["regression"].shape == (2, 5, 2)
    assert out["stain"].shape == (3, 2, SIZE, SIZE)
    with pytest.raises(ValueError, match="cond_dim"):
        TemporalEmbeddingModel(EMBED, RoPETransformerEncoder(D_MODEL), stain_head=ConditionedUNet(cond_dim=16))
    with pytest.raises(ValueError, match="Nothing reads"):
        TemporalEmbeddingModel(EMBED, RoPETransformerEncoder(D_MODEL), stain_head=ConditionedUNet())
    with pytest.raises(ValueError, match="wide"):
        model(torch.randn(1, 2, EMBED + 1))


@pytest.mark.parametrize("task", ["regression", "stain", "multitask", "unconditioned"])
def test_fit_and_test_write_scores(cellcycle_dir, tmp_path, task):
    """Every task variant trains, validates, and writes per-frame predictions and scores on the test split."""
    images = task != "regression"
    datamodule = make_datamodule(cellcycle_dir, images=images, embeddings=task != "unconditioned")
    module = TemporalEmbeddingModule(make_model(task), warmup_steps=1, output_dir=str(tmp_path / "out"))
    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=2,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(module, datamodule=datamodule)
    readouts = {"regression": ["regression"], "stain": ["stain"], "multitask": ["regression", "stain"]}
    for readout in readouts.get(task, ["stain"]):
        assert f"val/{readout}/pearson" in trainer.callback_metrics
        assert f"val/{readout}/phase_acc" in trainer.callback_metrics
    trainer.test(module, datamodule=datamodule)
    predictions = pd.read_csv(tmp_path / "out" / "predictions_test.csv")
    assert len(predictions) == 7  # the whole test track, every frame
    scores = json.loads((tmp_path / "out" / "metrics_test.json").read_text())
    assert set(scores) == set(readouts.get(task, ["stain"]))
    if images:
        assert predictions["stain_mko"].notna().all()


def test_a_perfect_prediction_scores_one():
    rng = np.random.default_rng(0)
    scale = np.array([1500.0, 2000.0])
    counts = np.stack([rng.uniform(0, 1500, 200), rng.uniform(0, 2000, 200)], axis=1)
    frames = pd.DataFrame({"track_id": np.repeat(np.arange(10), 20)})
    frames[["true_mko", "true_mvenus"]] = counts / scale
    frames[["pred_mko", "pred_mvenus"]] = counts / scale
    perfect = score_frames(frames, scale, THRESHOLDS)
    assert perfect["pearson"] == pytest.approx(1.0)
    assert perfect["track_pearson"] == pytest.approx(1.0)
    assert perfect["phase_acc"] == pytest.approx(1.0)
    assert perfect["mae"] == pytest.approx(0.0)
    # The truth's strict phase is unclear between the off and on levels; those frames are left out.
    assert perfect["n_phase_frames"] < len(frames)


def test_phase_rules_follow_pip_fucci():
    gem = np.array([20.0, 400.0, 400.0, 100.0])
    pip = np.array([500.0, 10.0, 500.0, 500.0])
    assert phase_strict(gem, pip, THRESHOLDS).tolist() == [0, 1, 2, -1]  # G1, S, G2/M, unclear
    assert phase_forced(gem, pip, THRESHOLDS).tolist() == [0, 1, 2, 2]


def test_masked_median_matches_numpy():
    images = torch.randn(4, 2, 9, 9)
    mask = torch.rand(4, 1, 9, 9) > 0.5
    mask[0] = False
    medians = masked_median(images, mask)
    assert torch.isnan(medians[0]).all()
    for i in range(1, 4):
        for c in range(2):
            values = images[i, c][mask[i, 0]].numpy()
            assert medians[i, c].item() == pytest.approx(float(np.sort(values)[(len(values) - 1) // 2]))


EXPERIMENTS = Path(__file__).resolve().parents[1] / "configs" / "cellcycle" / "experiments"
# Global token + centre patch mean of each encoder's cached embedding.
EMBED_DIMS = {
    "morphem": 384 + 384,
    "cell-dino": 1024 + 1024,
    "dinov3": 1024 + 1024,
    "c-radio": 2560 + 1280,
    "vit-untrained": 384 + 384,
}


@pytest.mark.parametrize("leaf", sorted(path.name for path in EXPERIMENTS.glob("*.yml")))
def test_experiment_configs_are_consistent(leaf):
    """Widths line up, images load exactly when a stain head exists, and checkpoints watch logged readouts."""
    config = load_composed_config(EXPERIMENTS / leaf)
    model = config["model"]["init_args"]["model"]["init_args"]
    data = config["data"]["init_args"]
    temporal, regression, stain = model.get("temporal"), model.get("regression_head"), model.get("stain_head")
    if temporal is not None:
        d_model = temporal["init_args"]["d_model"]
        if regression is not None:
            assert regression["init_args"]["in_dims"] == d_model
        if stain is not None:
            assert stain["init_args"]["cond_dim"] == d_model
        if data.get("phase_frames"):
            encoder = model["encoder"]["init_args"]
            assert data["embeddings"] is None and encoder["trainable"]
            name = encoder["name"]
        else:
            name = Path(data["embeddings"]).stem
        assert model["embed_dim"] == EMBED_DIMS[name]
    else:
        assert data["embeddings"] is None and stain["init_args"]["cond_dim"] == 0
    assert (data["images_per_batch"] > 0) == (stain is not None)
    monitors = {
        callback["init_args"]["monitor"]
        for callback in config["trainer"]["callbacks"]
        if callback["class_path"].endswith("ModelCheckpoint")
    }
    expected = {f"val/{name}/pearson" for name, head in (("regression", regression), ("stain", stain)) if head}
    assert monitors == expected


@pytest.mark.slow
@pytest.mark.parametrize("name", ["morphem", "cell-dino", "dinov3", "c-radio", "vit-untrained"])
def test_foundation_encoders_embed_phase_frames(name):
    """Each pretrained encoder maps phase frames to [global token | centre patch mean]."""
    from dynaformer.models import FoundationFrameEncoder

    encoder = FoundationFrameEncoder(name)
    with torch.no_grad():
        embedding = encoder(torch.randn(2, 1, 224, 224))
    assert embedding.shape == (2, encoder.token_dim + encoder.patch_dim)
    assert torch.isfinite(embedding).all()


@pytest.mark.slow
def test_untrained_vit_is_morphem_without_its_weights():
    """The untrained control has MorphEm's architecture, reproducible random weights, and none of MorphEm's.

    The trainable version starts from the same weights, so training is all that separates the two.
    """
    from dynaformer.models import FoundationFrameEncoder

    first, second = FoundationFrameEncoder("vit-untrained"), FoundationFrameEncoder("vit-untrained")
    trainable = FoundationFrameEncoder("vit-untrained", trainable=True)
    pretrained = FoundationFrameEncoder("morphem")
    for (name, a), b, t, c in zip(
        first.backbone.named_parameters(),
        second.backbone.parameters(),
        trainable.backbone.parameters(),
        pretrained.backbone.parameters(),
        strict=True,
    ):
        assert torch.equal(a, b) and torch.equal(a, t), name
        assert a.shape == c.shape, name
        assert t.requires_grad and not a.requires_grad, name
    weight = "model.blocks.0.attn.qkv.weight"
    untrained_weight = dict(first.backbone.named_parameters())[weight]
    assert not torch.allclose(untrained_weight, dict(pretrained.backbone.named_parameters())[weight])
    with pytest.raises(ValueError, match="can be trained"):
        FoundationFrameEncoder("dinov3", trainable=True)


@pytest.mark.slow
@pytest.mark.parametrize("trainable", [True, False])
def test_trained_encoder_learns_and_is_checkpointed(cellcycle_dir, tmp_path, trainable):
    """A trainable E learns and its weights stay in the checkpoint; a frozen one does neither."""
    from dynaformer.models import FoundationFrameEncoder

    def build() -> TemporalEmbeddingModule:
        encoder = FoundationFrameEncoder("vit-untrained", img_size=SIZE, center_patches=2, trainable=trainable)
        temporal = RoPETransformerEncoder(d_model=D_MODEL, n_heads=4, n_layers=1)
        model = TemporalEmbeddingModel(
            encoder.embed_dim, temporal=temporal, regression_head=MLPRegressionHead(D_MODEL), encoder=encoder
        )
        return TemporalEmbeddingModule(model, warmup_steps=1)

    module = build()
    initial = {k: v.clone() for k, v in module.model.encoder.state_dict().items()}
    datamodule = make_datamodule(cellcycle_dir, images=False, embeddings=False, phase_frames=True)
    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=0,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, datamodule=datamodule)
    assert module.model.encoder.training == trainable
    trained = module.model.encoder.state_dict()
    changed = [k for k in initial if not torch.equal(initial[k], trained[k])]
    assert bool(changed) == trainable
    trainer.save_checkpoint(tmp_path / "model.ckpt")
    state = torch.load(tmp_path / "model.ckpt", weights_only=False)["state_dict"]
    assert any(k.startswith("model.encoder.") for k in state) == trainable
    restored = build()
    restored.load_state_dict(state)
    for k, v in restored.model.encoder.state_dict().items():
        assert torch.equal(v, trained[k]), k
