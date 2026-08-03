"""Backfill both scalings of SSIM / NRMSE / PSNR into an existing pixel-metrics cache.

``compute_pixel_metrics`` reports each of SSIM, NRMSE and PSNR twice -- once
scale-sensitive (bare name, per-input min-max) and once scale-invariant
(``SI_`` prefix, least-squares affine fit to the target). Caches written before
that change carry only one of the two families under the bare names, and which
one depends on when the eval ran:

* before 2026-07-29 (``cdb45488``) the bare names were scale-SENSITIVE;
* after it they were scale-INVARIANT, under the same names.

Nothing on disk distinguishes the two, so the bare columns of a legacy cache
cannot be trusted without recomputation. This module recomputes both families
from the same arrays the eval read and merges them into the cached rows.

Why a separate pass instead of ``force_recompute.final_metrics=true``: pixel
metrics are ~4% of a warm-cache condition's compute (measured from
``eval_timing.csv``: 10.6 s of 253 s for an iPSC nucleus leaf, against 107 s of
per-timepoint feature pairwise + 71 s of MicroSSIM). A full final-metrics re-run
would also rewrite ``mask_metrics.csv`` and ``feature_metrics.csv``, moving
already-published KID / MicroSSIM numbers for no reason. This pass is a strict
column *add*: every other column, and both sibling CSVs, are left untouched.

The recomputation doubles as a provenance audit. Each cached bare column is
compared against both freshly computed families, so every condition is
classified ``scale_sensitive`` (pre-switch cache), ``scale_invariant``
(post-switch cache) or ``mismatch`` -- the last meaning the cached numbers
reproduce neither, i.e. the arrays on disk are no longer the ones that were
scored (a stale cache against a re-predicted store).

``PCC`` is the control and the write gate: it is affine-invariant, so a scaling
relabel cannot move it. When the cached PCC fails to reproduce, the cache was
produced from arrays other than the ones on disk now, and the condition is
reported ``stale_cache`` with **nothing written** -- adding recomputed pixel
columns there would leave them consistent with the current store while
``Spectral_PCC`` / ``MicroMS3IM`` and the mask/feature CSVs stayed consistent with
the old one. That leaf needs a real re-eval, not a column add.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from iohub.ngff import open_ome_zarr
from omegaconf import DictConfig, OmegaConf

from dynacell.evaluation._ref_hook import apply_dataset_ref
from dynacell.evaluation.metrics import compute_pixel_metrics

#: Bare (scale-sensitive) column -> scale-invariant counterpart.
SCALING_PAIRS: dict[str, str] = {"SSIM": "SI_SSIM", "NRMSE": "SI_NRMSE", "PSNR": "SI_PSNR"}

#: Columns this pass writes. ``PCC`` is recomputed too -- it is affine-invariant,
#: so it must be unchanged by the switch, which makes it the anchor that proves a
#: mismatch is a stale cache rather than a scaling relabel.
BACKFILL_COLUMNS: tuple[str, ...] = ("PCC", *SCALING_PAIRS, *SCALING_PAIRS.values())

#: Relative tolerance for "the cached column reproduces this family". Loose enough
#: to absorb GPU-vs-CPU float reduction order and a cubic patch release, tight
#: enough that the two families (which differ by ~40% on SSIM) never both match.
MATCH_RTOL = 1e-3
MATCH_ATOL = 1e-5


@dataclass
class ConditionReport:
    """Outcome of one condition's backfill."""

    name: str
    save_dir: Path
    status: str
    n_rows: int = 0
    provenance: dict[str, str] = field(default_factory=dict)
    detail: str = ""

    def line(self) -> str:
        """One-line summary for the run log."""
        prov = " ".join(f"{k}={v}" for k, v in sorted(self.provenance.items()))
        return f"[backfill] {self.status:<12} {self.name} ({self.n_rows} rows) {prov} {self.detail}".rstrip()


def _classify(cached: np.ndarray, sensitive: np.ndarray, invariant: np.ndarray) -> str:
    """Name the family a cached column reproduces.

    Compared on the finite entries only: a cache may legitimately hold NaN where a
    timepoint was degenerate, and an all-NaN column carries no provenance at all.
    """
    ok = np.isfinite(cached) & np.isfinite(sensitive) & np.isfinite(invariant)
    if not ok.any():
        return "empty"
    matches_sensitive = np.allclose(cached[ok], sensitive[ok], rtol=MATCH_RTOL, atol=MATCH_ATOL)
    matches_invariant = np.allclose(cached[ok], invariant[ok], rtol=MATCH_RTOL, atol=MATCH_ATOL)
    if matches_sensitive and matches_invariant:
        # The two families collapsed to one number -- possible on a synthetic
        # exactly-affine pair, never on real data. Reporting it as either would
        # assert a provenance the numbers do not support.
        return "degenerate"
    if matches_sensitive:
        return "scale_sensitive"
    if matches_invariant:
        return "scale_invariant"
    return "mismatch"


def _needed_timepoints(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Map FOV -> sorted timepoints, taken from the cached rows.

    Deriving the work list from the cache rather than re-walking the plates keeps
    this pass a pure column-add: it cannot silently score a different position set
    than the eval did (``limit_positions`` / ``exclude_fov_names`` / a partial
    prediction store are already baked into the rows).
    """
    by_fov: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_fov[str(row["FOV"])].append(int(row["Timepoint"]))
    return {fov: sorted(ts) for fov, ts in by_fov.items()}


def backfill_condition(config: DictConfig, *, force: bool = False, use_gpu: bool | None = None) -> ConditionReport:
    """Recompute both scalings for one eval condition and merge them into its cache.

    Parameters
    ----------
    config
        A fully merged single-condition eval config (``io.*``, ``save.*``,
        ``pixel_metrics.spacing`` resolved).
    force
        Recompute even when the cache already carries the ``SI_*`` columns.
    use_gpu
        Override ``config.use_gpu`` for the metric math. ``None`` follows the config.

    Returns
    -------
    ConditionReport
        ``status`` is ``written``, ``skipped`` (already backfilled), ``absent``
        (no pixel cache to add columns to), or ``stale_cache`` (the cached PCC does
        not reproduce, so nothing was written -- see the fail-closed note below).
    """
    name = str(OmegaConf.select(config, "name", default=Path(config.save.save_dir).name))
    save_dir = Path(config.save.save_dir)
    npy_path = save_dir / config.save.pixel_metrics_filename
    if not npy_path.is_file():
        return ConditionReport(name, save_dir, "absent", detail=f"no {npy_path.name}")

    rows: list[dict[str, Any]] = np.load(npy_path, allow_pickle=True).tolist()
    if not rows:
        return ConditionReport(name, save_dir, "absent", detail="empty pixel cache")
    if not force and all(col in rows[0] for col in SCALING_PAIRS.values()):
        return ConditionReport(name, save_dir, "skipped", n_rows=len(rows), detail="SI_* already present")

    gpu = bool(config.use_gpu if use_gpu is None else use_gpu)
    spacing = list(config.pixel_metrics.spacing)
    wanted = _needed_timepoints(rows)

    fresh: dict[tuple[str, int], dict[str, float]] = {}
    with (
        open_ome_zarr(Path(config.io.pred_path), mode="r") as pred_plate,
        open_ome_zarr(Path(config.io.gt_path), mode="r") as gt_plate,
    ):
        gt_by_name = dict(gt_plate.positions())
        for pos_name, pos_pred in pred_plate.positions():
            timepoints = wanted.get(pos_name)
            if timepoints is None:
                continue
            pos_gt = gt_by_name[pos_name]
            predict = np.asarray(pos_pred.data[:, pos_pred.get_channel_index(config.io.pred_channel_name)])
            target = np.asarray(pos_gt.data[:, pos_gt.get_channel_index(config.io.gt_channel_name)])
            for t in timepoints:
                scored = compute_pixel_metrics(
                    predict[t],
                    target[t],
                    spacing=spacing,
                    fsc_kwargs=None,
                    spectral_pcc_kwargs=None,
                    use_gpu=gpu,
                )
                # cubic returns 0-d cupy arrays on the GPU path; coerce here so the
                # comparison arrays below are plain numpy and the merged rows are
                # JSON/CSV-plain floats rather than device scalars.
                fresh[(pos_name, t)] = {k: float(v) for k, v in scored.items()}

    missing = sorted(set(_needed_timepoints(rows)) - {fov for fov, _ in fresh})
    if missing:
        raise KeyError(
            f"{name}: prediction store {config.io.pred_path} has no positions {missing!r}, "
            "but the cached pixel rows do -- the store was rewritten under the cache"
        )

    provenance: dict[str, str] = {}
    for bare, si in SCALING_PAIRS.items():
        cached = np.array([float(r.get(bare, np.nan)) for r in rows], dtype=float)
        sensitive = np.array([fresh[(str(r["FOV"]), int(r["Timepoint"]))][bare] for r in rows], dtype=float)
        invariant = np.array([fresh[(str(r["FOV"]), int(r["Timepoint"]))][si] for r in rows], dtype=float)
        provenance[bare] = _classify(cached, sensitive, invariant)
    cached_pcc = np.array([float(r.get("PCC", np.nan)) for r in rows], dtype=float)
    fresh_pcc = np.array([fresh[(str(r["FOV"]), int(r["Timepoint"]))]["PCC"] for r in rows], dtype=float)
    ok = np.isfinite(cached_pcc) & np.isfinite(fresh_pcc)
    provenance["PCC"] = (
        "reproduced"
        if ok.any() and np.allclose(cached_pcc[ok], fresh_pcc[ok], rtol=MATCH_RTOL, atol=MATCH_ATOL)
        else "MOVED"
    )
    if provenance["PCC"] == "MOVED":
        # Fail CLOSED. PCC is affine-invariant, so it cannot move under a scaling
        # relabel: the cache was produced from arrays other than the ones on disk
        # now. Writing the recomputed pixel columns would leave them consistent with
        # the current store while ``Spectral_PCC`` / ``MicroMS3IM`` / the mask and
        # feature CSVs stay consistent with the old one -- a mixed-provenance row,
        # which is the exact failure this module exists to prevent. Such a condition
        # needs a real re-eval (force_recompute.final_metrics=true), not a column add.
        worst = float(np.nanmax(np.abs(cached_pcc[ok] - fresh_pcc[ok]))) if ok.any() else float("nan")
        return ConditionReport(
            name,
            save_dir,
            "stale_cache",
            n_rows=len(rows),
            provenance=provenance,
            detail=f"cached PCC differs by up to {worst:.3g}; NOT written -- re-evaluate this leaf",
        )

    for row in rows:
        row.update({col: float(fresh[(str(row["FOV"]), int(row["Timepoint"]))][col]) for col in BACKFILL_COLUMNS})

    pd.DataFrame(rows).to_csv(save_dir / config.save.pixel_csv_filename, index=False)
    np.save(npy_path, rows)
    return ConditionReport(name, save_dir, "written", n_rows=len(rows), provenance=provenance)


def _conditions_of(config: DictConfig) -> list[DictConfig]:
    """Per-condition merged configs, or the single top-level config as a 1-list."""
    from dynacell.evaluation.pipeline import _merge_condition

    conditions = OmegaConf.select(config, "conditions", default=None)
    if not conditions:
        # A single-condition leaf still needs its manifest splice: ``io.*`` and
        # ``pixel_metrics.spacing`` come from ``benchmark.dataset_ref``, not from the
        # leaf. Applying it here rather than in the caller keeps the helper
        # self-contained -- a caller that only invoked _conditions_of would
        # otherwise hit MissingMandatoryValue on spacing.
        apply_dataset_ref(config)
        return [config]
    merged = []
    for idx, cond in enumerate(conditions):
        one = _merge_condition(config, cond)
        apply_dataset_ref(one)
        if OmegaConf.select(one, "name", default=None) is None:
            OmegaConf.update(one, "name", str(idx), force_add=True)
        merged.append(one)
    return merged


@hydra.main(version_base="1.2", config_path="_configs", config_name="eval")
def backfill_pixel_scalings(config: DictConfig) -> None:
    """Backfill both SSIM/NRMSE/PSNR scalings for every condition in a leaf.

    Takes the same ``leaf=grouped/<bucket>/eval_grouped`` override the grouped
    eval does, so the prediction/GT paths and spacing come from the one
    authoritative source instead of being re-derived here. Loads no models: only
    the pixel-metric math runs.

    Extra overrides
    ---------------
    ``+backfill.force=true``
        Recompute conditions that already carry the ``SI_*`` columns. Appended
        with ``+`` because the eval schema has no ``backfill`` block -- this is a
        backfill-only knob and does not belong in the eval contract.
    """
    force = bool(OmegaConf.select(config, "backfill.force", default=False))

    reports: list[ConditionReport] = []
    conditions = _conditions_of(config)
    for idx, one in enumerate(conditions, start=1):
        print(f"[backfill] ({idx}/{len(conditions)}) {one.save.save_dir}", flush=True)
        report = backfill_condition(one, force=force)
        print(report.line(), flush=True)
        reports.append(report)

    counts: dict[str, int] = defaultdict(int)
    for report in reports:
        counts[report.status] += 1
        for column, verdict in report.provenance.items():
            counts[f"{column}:{verdict}"] += 1
    print("[backfill] summary " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())), flush=True)

    stale = [r for r in reports if r.status == "stale_cache"]
    if stale:
        raise SystemExit(
            "cached PCC does not reproduce for "
            + ", ".join(str(r.save_dir) for r in stale)
            + " -- PCC is affine-invariant, so a scaling relabel cannot have moved it. Those caches "
            "were produced from other arrays than the stores hold now; nothing was written for them. "
            "Re-evaluate those leaves with force_recompute.final_metrics=true instead."
        )
