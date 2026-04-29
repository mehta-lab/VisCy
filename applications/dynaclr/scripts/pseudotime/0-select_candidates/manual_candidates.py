"""Write a manual productive-cohort CSV for debugging the pseudotime pipeline.

Hand-picked tracks organised as a nested dict keyed by
``(dataset_id, fov_name) → [track specs]``. Each track spec carries the
crop window ``[t_before, t_after]`` and a ``labels`` dict mapping each
label column to a list of ``[t_on, t_off]`` intervals (inclusive).

``t_key_event`` (the DTW anchor frame) is derived from
:data:`ANCHOR_LABEL` — the first positive frame of that label's
interval list. For ``infection_state`` this is the first ``infected``
frame.

Output schema matches :mod:`select_candidates`. The manual path is the
``productive`` cohort only; bystander/abortive/mock cohorts come from
:mod:`select_candidates` (auto path).

Run::

    uv run python manual_candidates.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(SCRIPT_DIR))
from lineage_utils import assign_lineage_ids, classify_divides  # noqa: E402

CANDIDATE_SET = "manual_debug_zikv"
ANCHOR_LABEL = "infection_state"
ANCHOR_POSITIVE = "infected"

# Output: same schema as select_candidates.py (productive cohort only).
ANNOTATIONS_OUTPUT = SCRIPT_DIR / "candidates" / f"{CANDIDATE_SET}_productive.csv"

# Label columns and their (positive, negative) values.
LABEL_VALUES: dict[str, tuple[str, str]] = {
    "infection_state": ("infected", "uninfected"),
    "organelle_state": ("remodeled", "noremodeled"),
    "cell_division_state": ("mitosis", "interphase"),
}

# Per-dataset metadata. ``embedding_zarr`` validates each spec against
# the embedding ``.obs`` and supplies ``parent_track_id`` when not given
# in the spec. ``transition_window_*_minutes`` set the divides classifier
# half-widths (must match candidates.yaml lineage_rules for the
# corresponding candidate set).
DATASETS: dict[str, dict] = {
    "2025_07_24_SEC61": {
        "frame_interval_minutes": 30.0,
        "embedding_zarr": (
            "/hpc/projects/organelle_phenotyping/models/DynaCLR-2D-MIP-BagOfChannels/"
            "2d-mip-ntxent-t0p2-lr2e5-bs256-192to160-zext11/evaluation_lc_v1/embeddings/"
            "2025_07_24_A549_viral_sensor_ZIKV.zarr"
        ),
        "transition_window_k_pre_minutes": 60.0,
        "transition_window_k_post_minutes": 120.0,
    },
}

# (dataset_id, fov_name) -> list of track specs.
# Each spec: track_id, t_before, t_after, labels.
# ``parent_track_id`` is optional; when missing, it's pulled from the
# embedding .obs during validation.
# ``labels`` maps label column -> list of [t_on, t_off] intervals (inclusive).
TRACKS: dict[tuple[str, str], list[dict]] = {
    ("2025_07_24_SEC61", "A/2/000000"): [
        {
            "track_id": 86,
            "t_before": 14,
            "t_after": 27,
            "labels": {"infection_state": [[20, 27]]},
        },
        {
            "track_id": 65,
            "t_before": 14,
            "t_after": 27,
            "labels": {"infection_state": [[19, 27]]},
        },
        {
            "track_id": 40,
            "t_before": 14,
            "t_after": 27,
            "labels": {"infection_state": [[19, 27]]},
        },
        {
            "track_id": 60,
            "t_before": 14,
            "t_after": 27,
            "labels": {"infection_state": [[19, 27]]},
        },
    ],
}


def _t_key_event(spec: dict) -> int:
    """First positive frame of ``ANCHOR_LABEL`` — the per-cell anchor."""
    labels = spec.get("labels", {})
    if ANCHOR_LABEL not in labels or not labels[ANCHOR_LABEL]:
        raise ValueError(f"Track {spec.get('track_id')!r} is missing an anchor interval for '{ANCHOR_LABEL}'.")
    return int(labels[ANCHOR_LABEL][0][0])


def _validate_against_anndata(dataset_id: str, fov_name: str, tracks: list[dict]) -> dict[int, int]:
    """Validate each track spec against the embedding ``.obs``.

    Returns a mapping ``track_id → parent_track_id`` from the embedding's
    obs so manual specs need not repeat the parent ids. Raises
    :class:`ValueError` listing every problem in one edit pass.
    """
    ds_meta = DATASETS[dataset_id]
    emb_path = ds_meta["embedding_zarr"]
    adata = ad.read_zarr(emb_path)
    adata.obs_names_make_unique()
    obs = adata.obs
    fov_obs = obs[obs["fov_name"].astype(str) == fov_name]

    problems: list[str] = []
    parent_lookup: dict[int, int] = {}

    for spec in tracks:
        tid = int(spec["track_id"])
        track_obs = fov_obs[fov_obs["track_id"].astype(int) == tid]
        if len(track_obs) == 0:
            problems.append(f"  {dataset_id} {fov_name} track_id={tid}: not found in {emb_path}")
            continue
        track_tps = set(track_obs["t"].astype(int).tolist())

        if "parent_track_id" in track_obs.columns:
            parent_lookup[tid] = int(track_obs.iloc[0]["parent_track_id"])
        else:
            parent_lookup[tid] = int(spec.get("parent_track_id", -1))

        for col in ("t_before", "t_after"):
            if int(spec[col]) not in track_tps:
                problems.append(
                    f"  {dataset_id} {fov_name} track_id={tid}: {col}={spec[col]} not in track "
                    f"(min={min(track_tps)}, max={max(track_tps)})"
                )

        labels = spec.get("labels", {})
        if ANCHOR_LABEL not in labels or not labels[ANCHOR_LABEL]:
            problems.append(f"  {dataset_id} {fov_name} track_id={tid}: missing anchor label '{ANCHOR_LABEL}'")
        for label_col, intervals in labels.items():
            if label_col not in LABEL_VALUES:
                problems.append(
                    f"  {dataset_id} {fov_name} track_id={tid}: unknown label column '{label_col}' "
                    f"(known: {list(LABEL_VALUES)})"
                )
            for interval in intervals:
                if len(interval) != 2 or interval[0] > interval[1]:
                    problems.append(
                        f"  {dataset_id} {fov_name} track_id={tid}: {label_col} interval {interval} "
                        f"is not a valid [t_on, t_off] pair"
                    )
                    continue
                if interval[0] not in track_tps or interval[1] not in track_tps:
                    problems.append(
                        f"  {dataset_id} {fov_name} track_id={tid}: {label_col} interval {interval} "
                        f"has frame(s) outside the track (min={min(track_tps)}, max={max(track_tps)})"
                    )

    if problems:
        raise ValueError("Manual candidate validation failed:\n" + "\n".join(problems))

    return parent_lookup


def build_annotation_rows() -> pd.DataFrame:
    """Expand every track spec into per-timepoint rows in the new schema."""
    rows = []
    parent_by_dataset: dict[str, dict[int, int]] = {}
    for (dataset_id, fov_name), tracks in TRACKS.items():
        parent_by_dataset[dataset_id] = _validate_against_anndata(dataset_id, fov_name, tracks)

    for (dataset_id, fov_name), tracks in TRACKS.items():
        parent_lookup = parent_by_dataset[dataset_id]
        for spec in tracks:
            tid = int(spec["track_id"])
            t_before = int(spec["t_before"])
            t_after = int(spec["t_after"])
            intervals = spec.get("labels", {})
            parent_tid = int(spec.get("parent_track_id", parent_lookup.get(tid, -1)))

            for t in range(t_before, t_after + 1):
                row = {
                    "dataset_id": dataset_id,
                    "fov_name": fov_name,
                    "track_id": tid,
                    "parent_track_id": parent_tid,
                    "t": t,
                }
                for label_col, (pos, neg) in LABEL_VALUES.items():
                    spans = intervals.get(label_col)
                    if spans is None:
                        row[label_col] = ""
                    elif any(lo <= t <= hi for lo, hi in spans):
                        row[label_col] = pos
                    else:
                        row[label_col] = neg
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    """Validate specs, reconnect lineages, classify divides, write CSV."""
    df = build_annotation_rows()
    df = assign_lineage_ids(df, return_both_branches=True)

    # Per-cell t_zero from the manual anchor.
    t_zero_lookup: dict[int, int] = {}
    for (dataset_id, fov_name), tracks in TRACKS.items():
        for spec in tracks:
            tid = int(spec["track_id"])
            mask = (df["dataset_id"] == dataset_id) & (df["fov_name"] == fov_name) & (df["track_id"] == tid)
            if not mask.any():
                continue
            lineage_id = int(df.loc[mask, "lineage_id"].iloc[0])
            if lineage_id < 0:
                continue
            # Set t_zero per lineage to the earliest manual anchor across
            # tracks in that lineage.
            t_anchor = _t_key_event(spec)
            t_zero_lookup[lineage_id] = min(t_zero_lookup.get(lineage_id, t_anchor), t_anchor)

    # Divides classification per lineage using its dataset's frame interval.
    divides: dict[int, str] = {}
    for lineage_id, lineage_df in df.groupby("lineage_id"):
        if lineage_id < 0:
            divides[lineage_id] = "none"
            continue
        t_zero = t_zero_lookup.get(int(lineage_id))
        if t_zero is None:
            divides[lineage_id] = "none"
            continue
        ds_id = str(lineage_df["dataset_id"].iloc[0])
        ds_meta = DATASETS[ds_id]
        frame_interval = float(ds_meta["frame_interval_minutes"])
        k_pre = int(round(float(ds_meta["transition_window_k_pre_minutes"]) / frame_interval))
        k_post = int(round(float(ds_meta["transition_window_k_post_minutes"]) / frame_interval))
        divides[lineage_id] = classify_divides(lineage_df, t_zero, k_pre, k_post)

    df["cohort"] = "productive"
    df["divides"] = df["lineage_id"].map(divides).fillna("none")

    output_columns = [
        "dataset_id",
        "fov_name",
        "lineage_id",
        "track_id",
        "parent_track_id",
        "t",
        "cohort",
        "divides",
        "infection_state",
        "organelle_state",
        "cell_division_state",
    ]
    out = df[output_columns]

    ANNOTATIONS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(ANNOTATIONS_OUTPUT, index=False)
    print(f"Wrote {len(out)} rows to {ANNOTATIONS_OUTPUT}")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
