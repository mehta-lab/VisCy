"""Causal sequence assembly for temporal classifiers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CausalSequences:
    """Row-aligned causal windows and context audit arrays."""

    sequences: np.ndarray
    full_context: np.ndarray
    distinct_observations: np.ndarray


def build_causal_sequences(
    labels: pd.DataFrame,
    features: np.ndarray,
    *,
    window: int = 5,
    group_columns: tuple[str, ...] | None = None,
    time_column: str = "t",
) -> CausalSequences:
    """Build left-padded windows ending at every target row.

    Early track positions repeat the first observed frame. The full_context flag
    is true only when all window frames exist and their integer timepoints are
    contiguous.
    """
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("features must have shape (rows, features)")
    if len(labels) != len(features):
        raise ValueError("labels and features must have the same length")
    if window < 1:
        raise ValueError("window must be at least 1")
    if time_column not in labels:
        raise KeyError(f"Missing time column: {time_column!r}")

    if group_columns is None:
        dataset_column = "dataset" if "dataset" in labels else "experiment"
        group_columns = (dataset_column, "fov_name", "track_id")
    missing = set(group_columns) - set(labels.columns)
    if missing:
        raise KeyError(f"Missing group columns: {sorted(missing)}")

    output = np.empty(
        (len(features), window, features.shape[1]),
        dtype=np.float32,
    )
    full = np.zeros(len(features), dtype=bool)
    distinct = np.zeros(len(features), dtype=np.int16)
    time = labels[time_column].to_numpy(int)
    groups = labels.groupby(list(group_columns), observed=True, sort=False).indices
    for track_rows in groups.values():
        rows = np.asarray(track_rows, dtype=int)
        rows = rows[np.argsort(time[rows], kind="stable")]
        for position, row in enumerate(rows):
            source_positions = np.clip(
                np.arange(position - window + 1, position + 1),
                0,
                len(rows) - 1,
            )
            selected = rows[source_positions]
            output[row] = features[selected]
            distinct[row] = len(np.unique(selected))
            full[row] = bool(len(np.unique(selected)) == window and np.all(np.diff(time[selected]) == 1))
    return CausalSequences(output, full, distinct)
