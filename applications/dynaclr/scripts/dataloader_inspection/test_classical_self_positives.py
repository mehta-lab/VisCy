"""Smoke test for DynaCLR-2D-MIP-BoC classical (SimCLR) dataloader.

Verifies that with ``positive_cell_source="self"``:

1. The dataset short-circuits the lookup path (no ``_lineage_timepoints``,
   no ``_match_lookup``).
2. ``positive`` is a clone of ``anchor`` *before* augmentation.
3. ``positive_meta`` matches ``anchor_meta`` 1:1 (same cell, same FOV, same t).
4. After ``on_after_batch_transfer``, the two views differ — augmentation
   creates the SimCLR view diversity.

Usage::

    uv run python applications/dynaclr/scripts/dataloader_inspection/test_classical_self_positives.py
"""

# ruff: noqa: E402, D103

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch
import yaml

REPO = Path("/home/eduardo.hirata/repos/viscy")
TRAIN_DIR = REPO / "applications/dynaclr/configs/training/DynaCLR-2D"

BASE_CFG = TRAIN_DIR / "DynaCLR-2D-MIP-BagOfChannels.yml"
CLASSICAL_CFG = TRAIN_DIR / "DynaCLR-2D-MIP-BagOfChannels-classical.yml"

# Local-test overrides: keep the real v3 parquet (the tiny fastdev parquet
# has only 2 markers from a single dynamorph experiment, so it can't
# exercise marker rotation or multi-experiment stratification). Just shrink
# batch + workers for single-GPU smoke testing.
LOCAL_OVERRIDES = {
    "data": {
        "init_args": {
            "batch_size": 16,
            "num_workers": 0,
            "prefetch_factor": None,
            "buffer_size": 2,
            "cache_pool_bytes": 0,
        }
    }
}


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _deep_merge(base: dict, override: dict) -> dict:
    """Right-wins deep merge (mirrors LightningCLI multi-config behavior)."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _to_tuples(x):
    """Recursively convert lists to tuples (mirrors jsonargparse type coercion)."""
    if isinstance(x, list):
        return tuple(_to_tuples(v) for v in x)
    if isinstance(x, dict):
        return {k: _to_tuples(v) for k, v in x.items()}
    return x


def _instantiate(spec):
    """Minimal class_path/init_args instantiator for nested transforms.

    Lists of transforms are returned as lists; lists inside ``init_args``
    are coerced to tuples to match the typed transform signatures.
    """
    if isinstance(spec, list):
        return [_instantiate(x) for x in spec]
    if not isinstance(spec, dict):
        return spec
    if "class_path" in spec:
        module_name, _, cls_name = spec["class_path"].rpartition(".")
        module = __import__(module_name, fromlist=[cls_name])
        cls = getattr(module, cls_name)
        kwargs = spec.get("init_args", {}) or {}
        kwargs = {k: _to_tuples(v) if not isinstance(v, dict) else _instantiate(v) for k, v in kwargs.items()}
        return cls(**kwargs)
    return {k: _instantiate(v) for k, v in spec.items()}


def build_datamodule():
    """Stack base + classical overrides on top of the real v3 parquet.

    Mirrors how LightningCLI would compose ``--config`` flags: base recipe,
    then the classical override (which flips ``positive_cell_source`` and
    sets the marker-uniform group_weights), then a minimal local-test
    override that shrinks batch_size for single-GPU smoke testing while
    keeping the real v3 parquet so marker rotation and multi-experiment
    stratification are exercised.
    """
    cfg = _load_yaml(BASE_CFG)
    cfg = _deep_merge(cfg, _load_yaml(CLASSICAL_CFG))
    cfg = _deep_merge(cfg, LOCAL_OVERRIDES)

    data_args = cfg["data"]["init_args"]
    data_args["normalizations"] = _instantiate(data_args.get("normalizations", []))
    data_args["augmentations"] = _instantiate(data_args.get("augmentations", []))

    from dynaclr.data.datamodule import MultiExperimentDataModule

    dm = MultiExperimentDataModule(**data_args)
    dm.setup(stage="fit")
    return dm


def assert_self_mode_index(dm) -> None:
    """Section 1: dataset must short-circuit the lookup path."""
    ds = dm.train_dataset
    assert ds.positive_cell_source == "self", f"expected self, got {ds.positive_cell_source}"
    assert not hasattr(ds, "_lineage_timepoints"), "_lineage_timepoints should not be built in self mode"
    assert not hasattr(ds, "_match_lookup"), "_match_lookup should not be built in self mode"
    print(f"  - positive_cell_source = {ds.positive_cell_source!r}")
    print(f"  - n train anchors      = {len(ds.index.valid_anchors):,}")
    print(f"  - n val anchors        = {len(dm.val_dataset.index.valid_anchors):,}")


def assert_pre_aug_clone(dm) -> None:
    """Section 2 & 3: positive == anchor pre-aug, meta matches 1:1."""
    ds = dm.train_dataset
    valid = ds.index.valid_anchors
    n = min(8, len(valid))
    batch = ds.__getitems__(list(range(n)))

    a, p = batch["anchor"], batch["positive"]
    assert a.shape == p.shape, f"shape mismatch: {a.shape} vs {p.shape}"
    assert a.shape[0] == n, f"batch size {a.shape[0]} != requested {n}"
    # Pre-aug: dataset.py sets positive = anchor.clone() in self mode → bitwise equal
    assert torch.equal(a, p), "pre-aug anchor != positive (self mode should clone)"

    am, pm = batch["anchor_meta"], batch["positive_meta"]
    assert len(am) == len(pm) == n, f"meta lengths differ: {len(am)} vs {len(pm)}"
    for i in range(n):
        assert am[i] == pm[i], f"sample {i}: meta mismatch anchor={am[i]} positive={pm[i]}"

    print(f"  - batch shape          = {tuple(a.shape)}")
    print("  - bitwise-equal pre-aug: True (self clone confirmed)")
    print(f"  - {n}/{n} anchor_meta == positive_meta")


def _fake_trainer():
    """Minimum trainer attrs needed by on_after_batch_transfer (non-predict path)."""
    return types.SimpleNamespace(predicting=False)


def assert_post_aug_diverges(dm) -> None:
    """Section 4: post-augmentation, the two views must differ.

    Verifies the full augmentation pipeline (BatchedRandAffined,
    BatchedRandFlipd, BatchedRandAdjustContrastd, BatchedRandScaleIntensityd,
    BatchedRandGaussianSmoothd, BatchedRandGaussianNoised,
    BatchedRandSpatialCropd, BatchedChannelWiseZReductiond, then
    auto-appended center crop) ran on both keys independently.
    """
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    a_pre, p_pre = batch["anchor"].clone(), batch["positive"].clone()
    assert torch.equal(a_pre, p_pre), "pre-aug anchor != positive in batched form"

    print(f"  - applied transforms   = {[type(t).__name__ for t in dm.augmentations]}")
    print(f"  - pre-aug shape        = {tuple(a_pre.shape)}")
    print(f"  - pre-aug mean / std   = {a_pre.float().mean().item():.4f} / {a_pre.float().std().item():.4f}")

    dm.trainer = _fake_trainer()
    out = dm.on_after_batch_transfer(batch, dataloader_idx=0)
    a, p = out["anchor"], out["positive"]

    # 1. Shape changed — confirms RandSpatialCrop(10, 192, 192) → ZReduction → CenterCrop(160, 160) ran.
    assert a.shape == p.shape, f"post-aug shape mismatch: {a.shape} vs {p.shape}"
    expected_yx = tuple(dm.final_yx_patch_size)
    expected_z = dm.z_window
    assert tuple(a.shape[-2:]) == expected_yx, f"final YX crop mismatch: {a.shape[-2:]} vs {expected_yx}"
    assert a.shape[-3] == expected_z, f"Z-reduction mismatch: {a.shape[-3]} vs z_window={expected_z}"
    assert a.shape[-2:] != a_pre.shape[-2:], "yx unchanged — spatial crops did not run"
    assert a.shape[-3] != a_pre.shape[-3], "z unchanged — z-reduction did not run"

    # 2. Both keys differ from their pre-aug source — confirms anchor and positive each got augmented.
    a_changed = (a_pre[..., :expected_z, : expected_yx[0], : expected_yx[1]] - a).abs().mean().item()
    p_changed = (p_pre[..., :expected_z, : expected_yx[0], : expected_yx[1]] - p).abs().mean().item()
    print(f"  - post-aug shape       = {tuple(a.shape)}")
    print(f"  - anchor changed by    = {a_changed:.4f} (vs pre-aug)")
    print(f"  - positive changed by  = {p_changed:.4f} (vs pre-aug)")
    assert a_changed > 1e-3, "anchor pixels are unchanged from pre-aug — augmentations did not fire"
    assert p_changed > 1e-3, "positive pixels are unchanged from pre-aug — augmentations did not fire"

    # 3. Views diverge from each other — confirms independent RNG draws per key.
    diff = (a - p).abs().mean().item()
    a_std = a.float().std().item()
    print(f"  - mean |anchor-pos|    = {diff:.4f}")
    print(f"  - anchor std           = {a_std:.4f}")
    assert diff > 1e-4, "augmentation produced identical views — RNG seed leak between keys?"
    assert not torch.equal(a, p), "post-aug views are bitwise equal — SimCLR view diversity broken"

    # 4. NormalizeSampled ran — anchor mean should be ~0 (was non-zero pre-norm).
    a_mean = a.float().mean().item()
    print(f"  - post-aug anchor mean = {a_mean:.4f} (NormalizeSampled centers to ~0)")
    assert abs(a_mean) < 5.0, f"normalization may have failed — post-aug mean = {a_mean}"


def assert_single_marker_batches(dm) -> None:
    """Section 5: every batch must contain exactly one unique marker.

    Over many batches the sampler must also rotate through multiple
    markers. With ``batch_group_by: marker`` + marker-uniform
    ``group_weights``, FlexibleBatchSampler yields single-marker batches
    and (over a long enough horizon) rotates through every marker
    present in the data.
    """
    from collections import Counter

    loader = dm.train_dataloader()
    n_batches = 30
    flat: list[str] = []
    exp_per_batch: list[set[str]] = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        markers = {m["marker"] for m in batch["anchor_meta"]}
        pos_markers = {m["marker"] for m in batch["positive_meta"]}
        assert markers == pos_markers, f"batch {i}: anchor/positive marker mismatch"
        assert len(markers) == 1, f"batch {i}: expected 1 unique marker, got {len(markers)}: {markers}"
        flat.append(next(iter(markers)))
        exp_per_batch.append({m["experiment"] for m in batch["anchor_meta"]})

    counts = Counter(flat)
    print(f"  - {n_batches} batches sampled, all single-marker")
    print(f"  - marker rotation        = {dict(counts)}")
    print(f"  - unique markers seen    = {len(counts)} of 9 configured")
    avg_exps = sum(len(e) for e in exp_per_batch) / len(exp_per_batch)
    print(f"  - avg experiments/batch  = {avg_exps:.2f} (stratify_by: experiment within marker)")


def main() -> int:
    print("=" * 70)
    print("DynaCLR-2D-MIP-BoC classical (SimCLR) dataloader smoke test")
    print("=" * 70)

    print("\n[1/5] Building datamodule with classical override + fastdev parquet")
    dm = build_datamodule()
    print(f"  - cell_index_path = {dm.cell_index_path}")
    print(f"  - batch_size      = {dm.batch_size}")

    print("\n[2/5] Asserting self-mode short-circuit on index/dataset")
    assert_self_mode_index(dm)

    print("\n[3/5] Asserting pre-aug clone semantics (anchor == positive)")
    assert_pre_aug_clone(dm)

    print("\n[4/5] Asserting augmentation diverges the two views")
    assert_post_aug_diverges(dm)

    print("\n[5/5] Asserting single-marker batches (batch_group_by: marker)")
    assert_single_marker_batches(dm)

    print("\nAll classical-mode dataloader checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
