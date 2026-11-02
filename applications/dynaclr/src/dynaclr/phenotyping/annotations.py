"""Observation-only annotation reads, snapshots and verified identity joins."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from dynaclr.phenotyping.config import Annotation, normalize_fov

KEYS = ["dataset", "fov", "track_id", "t"]


def read_annotations(sources: tuple[Annotation, ...]) -> tuple[pd.DataFrame, list[dict]]:
    """Read only observation tables and retain unmarked annotations as unreviewed."""
    import zarr
    from anndata.io import read_elem

    frames, identities = [], []
    for source in sources:
        # Deliberately never call anndata.read_zarr: X/obsm embeddings are irrelevant.
        group = zarr.open_group(source.path, mode="r")
        obs = read_elem(group["obs"])
        required = {"fov_name", "track_id", "t", "id", "empty"}
        if required - set(obs):
            raise ValueError(f"Annotation obs missing {required - set(obs)}: {source.path}")
        frame = obs.copy()
        frame["annotation_row_id"] = frame.index.astype(str)
        frame["source_dataset_label"] = frame.get("dataset", source.dataset)
        frame["dataset"] = source.dataset
        frame["fov"] = frame.fov_name.map(normalize_fov)
        for col in ("t", "track_id", "id"):
            values = pd.to_numeric(frame[col], errors="raise")
            if not np.isfinite(values).all() or (values != np.floor(values)).any():
                raise ValueError(f"Invalid annotation identifier {col}")
            frame[col] = values.astype(np.int64)
        empty = frame["empty"]
        marked = empty.astype("string").str.strip().str.lower().isin(["empty", "true", "1"])
        # Unmarked is unreviewed. Do not turn blank/False into signal-positive truth.
        frame["annotated_empty"] = pd.array(np.where(marked, True, None), dtype="boolean")
        frame["annotation_confidence"] = source.confidence
        frame["annotation_source"] = source.path
        frame["annotation_id"] = frame["id"]
        content_hash = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).hexdigest()
        identities.append(
            {
                "path": source.path,
                "dataset": source.dataset,
                "confidence": source.confidence,
                "rows": len(frame),
                "empty_rows": int(marked.sum()),
                "obs_sha256": content_hash,
            }
        )
        frames.append(frame.reset_index(drop=True))
    result = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            columns=KEYS + ["annotation_id", "annotated_empty", "annotation_confidence", "annotation_source"]
        )
    )
    if len(result) and result.duplicated(KEYS).any():
        raise ValueError("Annotation sources contain duplicate observation identities")
    return result, identities


def compare_annotations(annotations: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    """Retain not-processed, different-dataset and missing-cell annotation rows."""
    if observations.empty:
        result = annotations.copy()
        result["annotation_match_status"] = "not_processed"
        return result
    selected_cols = KEYS + [
        c
        for c in (
            "id",
            "marker",
            "channel_name",
            "assignment_status",
            "mean_intensity",
            "median_intensity",
            "p90_intensity",
            "valid_signal",
            "cell_id",
            "reason",
        )
        if c in observations
    ]
    result = annotations.merge(
        observations[selected_cols], how="left", on=KEYS, validate="one_to_many", suffixes=("_annotation", "")
    )
    observed = result["id"].notna()
    identity = observed & (result.annotation_id == result["id"])
    result["annotation_match_status"] = np.select(
        [~observed, observed & ~identity, identity & result.valid_signal.fillna(False)],
        ["not_processed", "identity_mismatch", "matched"],
        default="no_valid_cell",
    )
    # Identity mismatch must never supply fluorescence evidence for that annotation.
    for column in ("mean_intensity", "median_intensity", "p90_intensity"):
        if column in result:
            result.loc[~identity, column] = np.nan
    return result


def candidate_agreement(comparison: pd.DataFrame, thresholds: dict[str, tuple[float, ...]]) -> pd.DataFrame:
    """Compare diagnostic cutoffs with marked empty observations without activating filters."""
    records = []
    for (dataset, marker, confidence), group in (
        comparison.groupby(["dataset", "marker", "annotation_confidence"], dropna=False)
        if "marker" in comparison
        else []
    ):
        marked = group[group.annotated_empty.fillna(False)]
        valid = marked[marked.annotation_match_status == "matched"]
        for threshold in thresholds.get(marker, ()):
            records.append(
                {
                    "dataset": dataset,
                    "marker": marker,
                    "confidence": confidence,
                    "candidate_threshold": threshold,
                    "metric": "mean_intensity",
                    "annotated_empty_frames": len(marked),
                    "measured_empty_frames": len(valid),
                    "fraction_flagged": (float((valid.mean_intensity < threshold).mean()) if len(valid) else np.nan),
                    "threshold_status": "candidate_only_not_independent_validation",
                }
            )
    return pd.DataFrame(records)
