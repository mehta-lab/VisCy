"""Integration tests for ``lite_compare.py``.

A tiny source HCS store is subset by the real ``build_temporal_subset_zarr`` (spread
mode), and the lite store's provenance drives the comparison. The synthetic full eval
dir encodes each row's source timepoint in its metric values, so a lite row matched to
the wrong full frame shows up as a non-zero difference.

Run::

    uv run --no-sync pytest applications/dynacell/tools/lite_compare_test.py -q
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from build_temporal_subset_zarr import build_temporal_subset_zarr, spread_timepoints
from iohub.ngff import TransformationMeta, open_ome_zarr
from lite_compare import DEEP, dataset_level, frame_map, per_row

from dynacell.evaluation.feature_metrics import _kid, _median_cosine_similarity

_N_POS = 3
_T = 6
_N_KEEP = 2
_CELLS_PER_FRAME = 4


def _build_lite(tmp_path: Path) -> Path:
    """Write a ``_N_POS x _T`` source store and its spread-mode lite subset; return the lite path."""
    source = tmp_path / "full.zarr"
    with open_ome_zarr(source, layout="hcs", mode="w-", channel_names=["Phase3D", "Nuclei"], version="0.5") as plate:
        for i in range(_N_POS):
            pos = plate.create_position("0", "0", f"fov{i:04d}")
            data = np.arange(_T, dtype=np.float32)[:, None, None, None, None] * np.ones((_T, 2, 2, 4, 4), np.float32)
            pos.create_image("0", data, transform=[TransformationMeta(type="scale", scale=[1.0, 1.0, 1.0, 1.0, 1.0])])
    lite = tmp_path / "lite.zarr"
    build_temporal_subset_zarr(source, lite, channels=["Phase3D", "Nuclei"], mode="spread", n_timepoints=_N_KEEP)
    return lite


def _row_value(fov_ordinal: int, t_source: int) -> float:
    """Metric value of a full-benchmark row, unique per (FOV, source frame)."""
    return 0.1 * fov_ordinal + 0.01 * t_source


def _write_full_dir(path: Path) -> None:
    """Full eval dir: per-row CSVs over every (FOV, t) plus per-cell embeddings for each deep extractor."""
    rng = np.random.default_rng(0)
    (path / "embeddings").mkdir(parents=True)
    fovs = [f"0/0/fov{i:04d}" for i in range(_N_POS) for _ in range(_T)]
    tps = [t for _ in range(_N_POS) for t in range(_T)]
    vals = [_row_value(i, t) for i in range(_N_POS) for t in range(_T)]
    pd.DataFrame({"FOV": fovs, "Timepoint": tps, "SI_SSIM": vals}).to_csv(path / "pixel_metrics.csv", index=False)
    pd.DataFrame({"FOV": fovs, "Timepoint": tps, "Dice": vals}).to_csv(path / "mask_metrics.csv", index=False)
    cell_fov = np.repeat(fovs, _CELLS_PER_FRAME)
    cell_t = np.repeat(tps, _CELLS_PER_FRAME)
    for key in DEEP:
        gt = rng.normal(size=(len(cell_fov), 5)).astype(np.float32)
        pred = gt + 0.5 + 0.2 * rng.normal(size=gt.shape).astype(np.float32)
        for side, arr in (("gt", gt), ("pred", pred)):
            np.savez(
                path / f"embeddings/{side}_{key}_single_cell_embeddings.npz",
                embeddings=arr,
                fov=cell_fov,
                timepoint=cell_t,
            )


def _write_lite_dir(path: Path, full_dir: Path, fmap: dict[tuple[str, int], int]) -> None:
    """Lite eval dir whose rows and dataset KIDs equal the full benchmark at the mapped frames."""
    path.mkdir(parents=True)
    keys = sorted(fmap)
    fovs = [f for f, _ in keys]
    vals = [_row_value(int(f[-4:]), fmap[(f, t)]) for f, t in keys]
    rows = {"FOV": fovs, "Timepoint": [t for _, t in keys]}
    pd.DataFrame({**rows, "SI_SSIM": vals}).to_csv(path / "pixel_metrics.csv", index=False)
    pd.DataFrame({**rows, "Dice": vals}).to_csv(path / "mask_metrics.csv", index=False)
    kept = set(fmap.items())
    feats = {}
    for key, prefix in DEEP.items():
        sides = {}
        for side in ("gt", "pred"):
            with np.load(full_dir / f"embeddings/{side}_{key}_single_cell_embeddings.npz") as z:
                src = {(f, t) for (f, _), t in kept}
                sel = np.array([(str(f), int(t)) in src for f, t in zip(z["fov"], z["timepoint"])])
                sides[side] = z["embeddings"][sel]
        feats[f"Dataset_{prefix}_KID"] = _kid(sides["pred"], sides["gt"], 100, 1000, 2020)[0]
        feats[f"Dataset_{prefix}_Median_Cosine_Similarity"] = _median_cosine_similarity(sides["pred"], sides["gt"])
    pd.DataFrame([feats]).to_csv(path / "feature_metrics.csv", index=False)


def test_frame_map_reads_the_builders_source_timepoints(tmp_path: Path) -> None:
    """``(FOV, t_lite) -> t_source`` is exactly the spread rule the builder applied."""
    fmap = frame_map(_build_lite(tmp_path))
    expected = {
        (f"0/0/fov{i:04d}", t_lite): t_src
        for i in range(_N_POS)
        for t_lite, t_src in enumerate(spread_timepoints(i, _T, _N_KEEP))
    }
    assert fmap == expected
    # Discriminating: the mapping is not the identity for any position.
    assert any(t_lite != t_src for (_, t_lite), t_src in fmap.items())


def test_lite_rows_match_the_full_benchmark_at_the_mapped_frames(tmp_path: Path) -> None:
    """Per-row and dataset-level comparisons read zero difference through the provenance map."""
    fmap = frame_map(_build_lite(tmp_path))
    full_dir, lite_dir = tmp_path / "full_eval", tmp_path / "lite_eval"
    _write_full_dir(full_dir)
    _write_lite_dir(lite_dir, full_dir, fmap)

    rows = {r["metric"]: r for r in per_row(lite_dir, full_dir, fmap)}
    assert set(rows) == {"SI_SSIM", "Dice"}
    for r in rows.values():
        assert r["n_rows"] == _N_POS * _N_KEEP
        assert r["max_abs_diff"] == 0.0
        assert r["lite_mean"] == pytest.approx(r["full_restricted_mean"], abs=1e-15)

    ds = dataset_level(lite_dir, full_dir, fmap)
    assert len(ds) == 2 * len(DEEP)
    for r in ds:
        assert r["lite"] == pytest.approx(r["full_restricted"], rel=1e-6)
        assert r["n_gt_cells"] == _N_POS * _N_KEEP * _CELLS_PER_FRAME

    # Known positive: reading the lite frame index as the source frame is caught.
    identity = {k: k[1] for k in fmap}
    wrong = {r["metric"]: r for r in per_row(lite_dir, full_dir, identity)}
    assert wrong["SI_SSIM"]["max_abs_diff"] > 0.005
