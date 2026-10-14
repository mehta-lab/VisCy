"""Tests for the DynaFormer lineage sampler, head, and evaluator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from dynaclr.evaluation.temporal.config_dynaformer import DynaFormerConfig
from dynaclr.evaluation.temporal.dynaformer import _current_state_target, run_dynaformer
from dynaclr.evaluation.temporal.dynaformer_data import (
    EvalWindowSpec,
    FoldStandardizer,
    MixedMaskSampler,
    build_eval_windows,
    build_lineage_table,
    infer_frame_interval_hours,
)
from dynaclr.evaluation.temporal.dynaformer_model import (
    DynaFormerHead,
    HorizonDeltaMLP,
    anchor_states,
    batch_to_tensors,
    multi_horizon_infonce,
)
from dynaclr.evaluation.temporal.model import ContinuousRotaryEmbedding, RoPETransformerLayer

INTERVAL = 0.5


def _lineage_labels() -> pd.DataFrame:
    """Root track 1 (t 0-3) divides into 2 (t 4-7) and 3 (t 4-6); 3 divides into 4 (t 7-9)."""
    rows = []
    spec = {1: (range(0, 4), -1), 2: (range(4, 8), 1), 3: (range(4, 7), 1), 4: (range(7, 10), 3)}
    for tid, (times, parent) in spec.items():
        for t in times:
            rows.append(
                {
                    "experiment": "e",
                    "fov_name": "A/1/0",
                    "track_id": tid,
                    "parent_track_id": parent,
                    "t": t,
                    "hours_post_perturbation": 3.0 + INTERVAL * t,
                }
            )
    # a second, unrelated track in the same FOV supplies same-frame negatives
    for t in range(0, 10):
        rows.append(
            {
                "experiment": "e",
                "fov_name": "A/1/0",
                "track_id": 9,
                "parent_track_id": -1,
                "t": t,
                "hours_post_perturbation": 3.0 + INTERVAL * t,
            }
        )
    return pd.DataFrame(rows)


def test_lineage_paths_follow_parent_chain_with_division_boundaries() -> None:
    labels = _lineage_labels().iloc[::-1].reset_index(drop=True)
    table = build_lineage_table(labels)
    tid = labels["track_id"].to_numpy()
    chains = sorted(tuple(dict.fromkeys(tid[path])) for path in table.paths)
    assert chains == [(1, 2), (1, 3, 4), (9,)]
    for path in table.paths:
        assert np.all(np.diff(table.t[path]) > 0)
    boundary_tracks = set(tid[table.division_boundary])
    assert boundary_tracks == {2, 3, 4}
    first_frames = labels.groupby("track_id")["t"].idxmin()
    assert set(np.flatnonzero(table.division_boundary)) == {first_frames[2], first_frames[3], first_frames[4]}
    assert table.multiplicity[labels.index[tid == 1]].tolist() == [2, 2, 2, 2]
    assert table.stats["n_tracks_with_parent"] == 3


def test_frame_interval_is_inferred_and_nonuniform_rejected() -> None:
    labels = _lineage_labels()
    assert infer_frame_interval_hours(labels) == pytest.approx(INTERVAL)
    bad = labels.copy()
    bad.loc[bad.index[-1], "hours_post_perturbation"] += 0.25
    with pytest.raises(ValueError, match="not uniform"):
        infer_frame_interval_hours(bad)


def test_sampler_covers_lengths_ends_and_never_shows_targets() -> None:
    labels = _lineage_labels()
    table = build_lineage_table(labels)
    features = np.random.default_rng(0).normal(size=(len(labels), 6)).astype(np.float32)
    sampler = MixedMaskSampler(
        table,
        features,
        allowed_rows=np.ones(len(labels), bool),
        k_frames=[1, 2],
        n_max_frames=6,
        frame_dropout=0.2,
        causal_probability=0.5,
        n_negatives=3,
        seed=1,
    )
    seen_length_one = seen_start = seen_end = False
    for _ in range(40):
        batch = sampler.sample_batch(50)
        lengths = (batch["window_rows"] >= 0).sum(1)
        seen_length_one |= bool((lengths == 1).any())
        seen_start |= bool((batch["p_index"] == 0).any())
        seen_end |= bool((batch["p_index"] == lengths - 1).any())
        for b in range(50):
            rows = batch["window_rows"][b]
            anchor_slot = batch["p_index"][b]
            assert batch["visible"][b, anchor_slot]
            assert rows[anchor_slot] == batch["anchor_row"][b]
            for j in range(2):
                if batch["target_valid"][b, j]:
                    target = batch["target_row"][b, j]
                    slots = np.flatnonzero(rows == target)
                    assert not batch["visible"][b, slots].any()
                    assert table.t[target] == table.t[batch["anchor_row"][b]] + sampler.k_frames[j]
                    negatives = batch["negative_row"][b, j]
                    negatives = negatives[negatives >= 0]
                    assert target not in negatives
                    assert np.all(table.t[negatives] == table.t[target])
            if batch["is_causal"][b]:
                assert not batch["visible"][b, anchor_slot + 1 :].any()
    assert seen_length_one and seen_start and seen_end


def test_eval_windows_no_lineage_and_hidden_offsets() -> None:
    labels = _lineage_labels()
    table = build_lineage_table(labels)
    tid = labels["track_id"].to_numpy()
    row = int(np.flatnonzero((tid == 4) & (labels["t"].to_numpy() == 8))[0])
    full = build_eval_windows(table, np.array([row]), EvalWindowSpec(history_budget=32))
    rows = full["window_rows"][0][full["window_rows"][0] >= 0]
    assert set(tid[rows]) == {1, 3, 4}
    own = build_eval_windows(table, np.array([row]), EvalWindowSpec(history_budget=32, no_lineage=True))
    rows = own["window_rows"][0][own["window_rows"][0] >= 0]
    assert set(tid[rows]) == {4}
    row = int(np.flatnonzero((tid == 1) & (labels["t"].to_numpy() == 1))[0])
    bidir = build_eval_windows(
        table, np.array([row]), EvalWindowSpec(history_budget=4, future_budget=6, hide_offsets=(1, 2))
    )
    offsets = table.t[bidir["window_rows"][0]] - table.t[row]
    visible = bidir["visible"][0]
    assert not visible[np.isin(offsets, (1, 2))].any()
    assert visible[offsets == 3].all()


def test_continuous_rope_is_invariant_to_global_time_shift() -> None:
    torch.manual_seed(0)
    layer = RoPETransformerLayer(16, 2, 2, 0.0, rope=ContinuousRotaryEmbedding(8, 0.2, 50.0)).eval()
    x = torch.randn(2, 5, 16)
    hours = torch.tensor([[3.0, 3.5, 4.0, 5.0, 7.5]] * 2)
    mask = torch.ones(2, 5, dtype=torch.bool)
    a = layer(x, positions=hours, key_padding_mask=mask)
    b = layer(x, positions=hours + 7.3, key_padding_mask=mask)
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    c = layer(x, positions=hours * 2.0, key_padding_mask=mask)
    assert not torch.allclose(a, c, atol=1e-3)


def test_infonce_loss_falls_on_planted_drift() -> None:
    labels = _lineage_labels()
    table = build_lineage_table(labels)
    rng = np.random.default_rng(3)
    drift = rng.normal(size=6).astype(np.float32)
    features = (labels["t"].to_numpy()[:, None] * drift + rng.normal(scale=0.1, size=(len(labels), 6))).astype(
        np.float32
    )
    features[labels["track_id"].to_numpy() == 9] *= -1.0
    standardizer = FoldStandardizer.fit(features, np.arange(len(labels)))
    feat = torch.from_numpy(standardizer.apply(features))
    sampler = MixedMaskSampler(
        table,
        feat.numpy(),
        allowed_rows=np.ones(len(labels), bool),
        k_frames=[1, 2],
        n_max_frames=6,
        frame_dropout=0.0,
        causal_probability=0.5,
        n_negatives=4,
        seed=2,
    )
    torch.manual_seed(0)
    head = DynaFormerHead(6, d_model=16, n_heads=2, n_layers=1, feedforward_multiplier=2, dropout=0.0)
    mlp = HorizonDeltaMLP(16, 2, 6)
    opt = torch.optim.Adam(list(head.parameters()) + list(mlp.parameters()), lr=3e-3)
    losses = []
    for _ in range(120):
        batch = batch_to_tensors(sampler.sample_batch(32), torch.device("cpu"))
        h = anchor_states(head, batch, feat)
        pred = feat[batch["anchor_row"]][:, None] + mlp(h)
        loss, _, _ = multi_horizon_infonce(
            pred,
            feat[batch["target_row"].clamp_min(0)],
            batch["target_valid"],
            feat[batch["negative_row"].clamp_min(0)],
            batch["negative_row"] >= 0,
            0.1,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert np.mean(losses[-20:]) < np.mean(losses[:20])


def test_current_state_target_leaves_missing_annotations_nan(tmp_path) -> None:
    labels = pd.DataFrame({"infection_state": ["uninfected", None, " Infected ", np.nan, "unknown"]})
    y = _current_state_target(labels, _smoke_config(tmp_path))
    np.testing.assert_array_equal(y, np.array([0.0, np.nan, 1.0, np.nan, np.nan], dtype=np.float32))


def _smoke_config(tmp_path, n_folds: int = 2) -> DynaFormerConfig:
    return DynaFormerConfig(
        datasets=[
            {"experiment": "e1", "annotation": str(tmp_path / "e1.csv")},
            {"experiment": "e2", "annotation": str(tmp_path / "e2.csv")},
        ],
        events=[{"name": "infection", "column": "infection_state", "negative": "uninfected", "positive": "infected"}],
        k_frames=[1, 2],
        horizons_hours=[0.5, 1.0],
        n_max_frames=6,
        history_budget=[1, 3, 6],
        future_budget=[0, 2],
        n_folds=n_folds,
        head={"d_model": 16, "n_heads": 2, "n_layers": 1},
        train={"steps": 30, "batch_size": 32, "n_negatives": 4, "log_every": 10},
        probe={"last_k": [2, 3]},
    )


def _write_smoke_stores(tmp_path) -> list:
    import anndata as ad

    rng = np.random.default_rng(5)
    paths = []
    for experiment in ("e1", "e2"):
        rows = []
        for fov in range(4):
            for track in range(6):
                onset = rng.integers(4, 12)
                parent = -1 if track < 4 else track - 4
                # tracks 0 and 1 divide at t=6 into 4 and 5; parents end where daughters start
                stop = 6 if track < 2 else 14
                for t in range(0 if parent < 0 else 6, stop):
                    state = "infected" if (track % 2 == 0 and t >= onset) else "uninfected"
                    rows.append(
                        {
                            "experiment": experiment,
                            "fov_name": f"A/{fov}/0",
                            "track_id": track,
                            "parent_track_id": parent,
                            "t": t,
                            "hours_post_perturbation": 3.0 + 0.5 * t,
                            "infection_state": state,
                            "is_blank": False,
                        }
                    )
        obs = pd.DataFrame(rows)
        signal = (obs["infection_state"] == "infected").to_numpy()[:, None] * 2.0
        X = (rng.normal(size=(len(obs), 8)) + signal + obs["t"].to_numpy()[:, None] * 0.1).astype(np.float32)
        adata = ad.AnnData(X=X, obs=obs.drop(columns=["infection_state"]))
        adata.obs_names = [str(i) for i in range(len(obs))]
        path = tmp_path / experiment / "Phase3D.zarr"
        path.parent.mkdir(parents=True)
        adata.write_zarr(path)
        obs[["fov_name", "track_id", "t", "infection_state"]].to_csv(tmp_path / f"{experiment}.csv", index=False)
        paths.append(path)
    return paths


def test_run_dynaformer_smoke_writes_all_outputs(tmp_path) -> None:
    paths = _write_smoke_stores(tmp_path)
    config = _smoke_config(tmp_path)
    out = tmp_path / "out"
    info = run_dynaformer(paths, config, out)
    for name in (
        "probe_metrics.csv",
        "forecast_skill.csv",
        "value_of_history.csv",
        "value_of_future.csv",
        "time_shuffle.csv",
        "rank_gate.csv",
        "onset_timing.csv",
        "target_audit.csv",
        "training_log.csv",
        "out_of_fold_probabilities.parquet",
        "fold_manifest.parquet",
        "run_info.json",
        "_SUCCESS.json",
    ):
        assert (out / name).exists(), name
    assert info["frame_interval_hours"] == pytest.approx(0.5)
    assert info["lineage"]["n_tracks_with_parent"] > 0
    assert info["lineage"]["n_orphan_singletons"] == 0
    assert info["lineage"]["n_rows_dropped_nonmonotonic"] == 0
    skill = pd.read_csv(out / "forecast_skill.csv")
    assert {"head", "persistence", "linear_extrapolation"} <= set(skill["predictor"])
    probes = pd.read_csv(out / "probe_metrics.csv")
    assert {"z", "h_causal", "h_bidir", "z_concat_k2", "h_causal_nolineage"} <= set(probes["representation"])
    assert set(probes["task"]) == {"onset_within", "current_state"}
    history = pd.read_csv(out / "value_of_history.csv")
    assert sorted(history["history_budget"].unique()) == [1, 3, 6]
