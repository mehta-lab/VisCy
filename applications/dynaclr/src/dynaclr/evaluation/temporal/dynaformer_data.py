"""Lineage-path sequences, mixed-mask sampling, and raw-sequence baselines.

A sequence is one root-to-leaf lineage path built per ``(experiment, fov_name)``
from ``track_id`` and ``parent_track_id``. A cell's past is therefore unique
(its own frames plus its ancestors'), while its future follows one daughter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from dynaclr.evaluation.temporal.forecast import _canonical_tokens


@dataclass(frozen=True)
class LineageTable:
    """Row-aligned lineage structure over a pooled label table.

    Attributes
    ----------
    paths : list of np.ndarray
        Row indices of each root-to-leaf path, strictly increasing in time.
    hours, t, track_id, group_token : np.ndarray
        Per-row time in hours, integer frame, track id, and ``experiment/fov`` key.
    division_boundary : np.ndarray
        Per-row flag, true on the first frame of a track whose parent is present.
    canonical_path, canonical_position : np.ndarray
        For each row, the first path containing it and its position there.
    multiplicity : np.ndarray
        Number of paths containing each row (ancestors are shared).
    frame_index : dict
        ``(group_token, t)`` to the rows observed at that frame.
    stats : dict
        Lineage construction statistics for ``run_info.json``.
    """

    paths: list[np.ndarray]
    hours: np.ndarray
    t: np.ndarray
    track_id: np.ndarray
    group_token: np.ndarray
    division_boundary: np.ndarray
    canonical_path: np.ndarray
    canonical_position: np.ndarray
    multiplicity: np.ndarray
    frame_index: dict[tuple[str, int], np.ndarray] = field(repr=False)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return len(self.hours)

    def path_of(self, row: int) -> tuple[np.ndarray, int]:
        """Return the canonical path of ``row`` and the row's position in it."""
        return self.paths[self.canonical_path[row]], int(self.canonical_position[row])


def infer_frame_interval_hours(
    labels: pd.DataFrame,
    *,
    time_column: str = "hours_post_perturbation",
    tolerance: float = 1e-6,
) -> float:
    """Return the single frame interval in hours, asserting it is uniform.

    Raises
    ------
    ValueError
        If ``hours`` is not an affine function of the integer frame ``t`` with one
        shared slope across all experiments.
    """
    hours = labels[time_column].to_numpy(dtype=float)
    t = labels["t"].to_numpy(dtype=float)
    experiment = labels["experiment"].astype(str).to_numpy()
    slopes: dict[str, list[float]] = {}
    groups = labels.groupby(["experiment", "fov_name", "track_id"], observed=True, sort=False).indices
    for rows in groups.values():
        rows = np.asarray(rows)
        order = np.argsort(t[rows], kind="stable")
        tt, hh = t[rows][order], hours[rows][order]
        keep = np.diff(tt) > 0
        if not keep.any():
            continue
        local = np.diff(hh)[keep] / np.diff(tt)[keep]
        slopes.setdefault(experiment[rows[0]], []).extend(local.tolist())
    if not slopes:
        raise ValueError("Could not infer a frame interval (no multi-frame tracks)")
    per_experiment = []
    for name, values in slopes.items():
        if np.ptp(values) > tolerance:
            raise ValueError(f"{name}: frame interval is not uniform within the experiment")
        per_experiment.append(float(np.median(values)))
    if np.ptp(per_experiment) > tolerance:
        raise ValueError(f"Frame interval differs across experiments: {per_experiment}")
    slopes = per_experiment  # type: ignore[assignment]
    return float(np.median(slopes))


def build_lineage_table(
    labels: pd.DataFrame,
    *,
    parent_column: str = "parent_track_id",
    time_column: str = "hours_post_perturbation",
    group_columns: Sequence[str] = ("experiment", "fov_name"),
) -> LineageTable:
    """Assemble root-to-leaf lineage paths per field of view.

    Falls back to one path per track when ``parent_column`` is absent. Rows whose
    frame does not strictly increase along a path (tracking overlap at a division)
    are dropped from that path only; rows left in no path get a singleton path.
    """
    hours = labels[time_column].to_numpy(dtype=float)
    t = labels["t"].to_numpy(dtype=int)
    track_id = labels["track_id"].to_numpy(dtype=int)
    group_token = _canonical_tokens(labels, group_columns)
    has_parent = parent_column in labels.columns
    parent_values = (
        pd.to_numeric(labels[parent_column], errors="coerce").fillna(-1).to_numpy(dtype=int)
        if has_parent
        else np.full(len(labels), -1, dtype=int)
    )

    paths: list[np.ndarray] = []
    division_boundary = np.zeros(len(labels), dtype=bool)
    n_tracks = n_linked = n_cycles = n_dropped = 0
    groups = labels.groupby(list(group_columns), observed=True, sort=False).indices
    for raw_rows in groups.values():
        rows = np.asarray(raw_rows, dtype=int)
        by_track: dict[int, np.ndarray] = {}
        for tid, positions in pd.Series(np.arange(len(rows))).groupby(track_id[rows]).indices.items():
            local = rows[np.asarray(positions, dtype=int)]
            by_track[int(tid)] = local[np.argsort(t[local], kind="stable")]
        n_tracks += len(by_track)
        parent_of: dict[int, int] = {}
        for tid, track_rows in by_track.items():
            candidates = parent_values[track_rows]
            candidates = candidates[candidates >= 0]
            if len(candidates):
                parent = int(candidates[0])
                if parent in by_track and parent != tid:
                    parent_of[tid] = parent
        children = set(parent_of.values())
        n_linked += len(parent_of)
        for tid in parent_of:
            division_boundary[by_track[tid][0]] = True
        for leaf in by_track:
            if leaf in children:
                continue
            chain = [leaf]
            visited = {leaf}
            current = leaf
            while current in parent_of:
                current = parent_of[current]
                if current in visited:
                    n_cycles += 1
                    break
                visited.add(current)
                chain.append(current)
            chain.reverse()
            path_rows = np.concatenate([by_track[tid] for tid in chain])
            keep = np.ones(len(path_rows), dtype=bool)
            running = -np.inf
            for index, row in enumerate(path_rows):
                if t[row] <= running:
                    keep[index] = False
                else:
                    running = t[row]
            n_dropped += int((~keep).sum())
            paths.append(path_rows[keep])

    multiplicity = np.zeros(len(labels), dtype=np.int32)
    for path in paths:
        multiplicity[path] += 1
    orphans = np.flatnonzero(multiplicity == 0)
    for row in orphans:
        paths.append(np.array([row], dtype=int))
        multiplicity[row] = 1

    canonical_path = np.full(len(labels), -1, dtype=np.int64)
    canonical_position = np.full(len(labels), -1, dtype=np.int64)
    for path_index, path in enumerate(paths):
        unassigned = canonical_path[path] < 0
        canonical_path[path[unassigned]] = path_index
        canonical_position[path[unassigned]] = np.flatnonzero(unassigned)
    if (canonical_path < 0).any():
        raise RuntimeError("Some rows were not assigned to any lineage path")

    frame_index: dict[tuple[str, int], np.ndarray] = {}
    frame_frame = pd.DataFrame({"g": group_token, "t": t})
    for key, rows in frame_frame.groupby(["g", "t"], observed=True, sort=False).indices.items():
        frame_index[(str(key[0]), int(key[1]))] = np.asarray(rows, dtype=int)

    lengths = np.array([len(path) for path in paths])
    track_lengths = labels.groupby([*group_columns, "track_id"], observed=True).size().to_numpy()
    stats = {
        "lineage_linking": "parent_track_id" if has_parent else "none (tracks only)",
        "n_tracks": int(n_tracks),
        "n_tracks_with_parent": int(n_linked),
        "n_paths": int(len(paths)),
        "n_cycles_broken": int(n_cycles),
        "n_rows_dropped_nonmonotonic": int(n_dropped),
        "n_orphan_singletons": int(len(orphans)),
        "path_length_quantiles_10_50_90": [float(q) for q in np.percentile(lengths, [10, 50, 90])],
        "track_length_quantiles_10_50_90": [float(q) for q in np.percentile(track_lengths, [10, 50, 90])],
        "row_multiplicity_mean": float(multiplicity.mean()),
    }
    return LineageTable(
        paths=paths,
        hours=hours,
        t=t,
        track_id=track_id,
        group_token=group_token,
        division_boundary=division_boundary,
        canonical_path=canonical_path,
        canonical_position=canonical_position,
        multiplicity=multiplicity,
        frame_index=frame_index,
        stats=stats,
    )


@dataclass
class FoldStandardizer:
    """Per-feature standardization fitted on unique training-fold rows."""

    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray, rows: np.ndarray) -> "FoldStandardizer":
        unique_rows = np.unique(np.asarray(rows, dtype=int))
        subset = features[unique_rows]
        center = subset.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = np.maximum(subset.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6)
        return cls(center=center, scale=scale)

    def apply(self, features: np.ndarray) -> np.ndarray:
        return ((features - self.center) / self.scale).astype(np.float32)


def window_aux_features(hours: np.ndarray, visible: np.ndarray) -> np.ndarray:
    """Elapsed hours to the previous *visible* frame; zero for the first one.

    Parameters
    ----------
    hours : np.ndarray
        Shape ``(B, T)``.
    visible : np.ndarray
        Boolean ``(B, T)``.
    """
    delta = np.zeros_like(hours, dtype=np.float32)
    for b in range(hours.shape[0]):
        idx = np.flatnonzero(visible[b])
        if len(idx) > 1:
            delta[b, idx[1:]] = hours[b, idx[1:]] - hours[b, idx[:-1]]
    return delta


class MixedMaskSampler:
    """Draw training windows along lineage paths with mixed attention masks.

    Every draw picks an anchor position ``p`` on a path, a window of random length
    containing it, a causal or bidirectional visibility pattern, random frame
    dropout, and hides the prediction-target frames ``p + k`` from the context.
    Negatives for each horizon are other cells observed in the same field of view
    at the target frame.
    """

    def __init__(
        self,
        table: LineageTable,
        features: np.ndarray,
        *,
        allowed_rows: np.ndarray,
        k_frames: Sequence[int],
        n_max_frames: int,
        frame_dropout: float,
        causal_probability: float,
        n_negatives: int,
        seed: int,
    ) -> None:
        self.table = table
        self.features = features
        self.k_frames = np.asarray(k_frames, dtype=int)
        self.n_max = int(n_max_frames)
        self.frame_dropout = float(frame_dropout)
        self.causal_probability = float(causal_probability)
        self.n_negatives = int(n_negatives)
        self.rng = np.random.default_rng(seed)
        allowed = np.asarray(allowed_rows, dtype=bool)
        anchor_path: list[int] = []
        anchor_pos: list[int] = []
        for path_index, path in enumerate(table.paths):
            if not allowed[path].all():
                if allowed[path].any():
                    raise ValueError("A lineage path straddles the train/test split")
                continue
            t_path = table.t[path]
            for position in range(len(path)):
                targets = t_path[position] + self.k_frames
                if np.isin(targets, t_path).any():
                    anchor_path.append(path_index)
                    anchor_pos.append(position)
        if not anchor_path:
            raise ValueError("No anchor has a valid prediction target in the training rows")
        self.anchor_path = np.asarray(anchor_path)
        self.anchor_pos = np.asarray(anchor_pos)

    @property
    def n_anchors(self) -> int:
        return len(self.anchor_path)

    def sample_batch(self, batch_size: int) -> dict[str, np.ndarray]:
        """Return one padded batch as numpy arrays."""
        table, rng = self.table, self.rng
        K, N, L_max = len(self.k_frames), self.n_negatives, self.n_max
        window_rows = np.full((batch_size, L_max), -1, dtype=np.int64)
        hours = np.zeros((batch_size, L_max), dtype=np.float32)
        division = np.zeros((batch_size, L_max), dtype=np.float32)
        visible = np.zeros((batch_size, L_max), dtype=bool)
        p_index = np.zeros(batch_size, dtype=np.int64)
        anchor_row = np.zeros(batch_size, dtype=np.int64)
        target_row = np.full((batch_size, K), -1, dtype=np.int64)
        target_valid = np.zeros((batch_size, K), dtype=bool)
        target_crosses_division = np.zeros((batch_size, K), dtype=bool)
        negative_row = np.full((batch_size, K, N), -1, dtype=np.int64)
        is_causal = np.zeros(batch_size, dtype=bool)

        picks = rng.integers(0, self.n_anchors, size=batch_size)
        for b, pick in enumerate(picks):
            path = table.paths[self.anchor_path[pick]]
            p = int(self.anchor_pos[pick])
            n = len(path)
            t_path = table.t[path]
            L = int(rng.integers(1, min(self.n_max, n) + 1))
            start = int(rng.integers(max(0, p - L + 1), min(p, n - L) + 1))
            window = path[start : start + L]
            vis = np.ones(L, dtype=bool)
            causal = rng.random() < self.causal_probability
            if causal:
                vis[p - start + 1 :] = False
            drop = rng.random(L) < self.frame_dropout
            drop[p - start] = False
            vis &= ~drop
            anchor = path[p]
            group = table.group_token[anchor]
            for j, k in enumerate(self.k_frames):
                q = int(np.searchsorted(t_path, t_path[p] + k))
                if q < n and t_path[q] == t_path[p] + k:
                    target = path[q]
                    target_row[b, j] = target
                    target_valid[b, j] = True
                    target_crosses_division[b, j] = table.track_id[target] != table.track_id[anchor]
                    if start <= q < start + L:
                        vis[q - start] = False
                    pool = table.frame_index.get((str(group), int(t_path[p] + k)))
                    if pool is not None:
                        pool = pool[pool != target]
                        if len(pool):
                            chosen = rng.choice(pool, size=min(N, len(pool)), replace=False)
                            negative_row[b, j, : len(chosen)] = chosen
            window_rows[b, :L] = window
            hours[b, :L] = table.hours[window]
            hours[b, L:] = table.hours[anchor]
            division[b, :L] = table.division_boundary[window]
            visible[b, :L] = vis
            p_index[b] = p - start
            anchor_row[b] = anchor
            is_causal[b] = causal
        return {
            "window_rows": window_rows,
            "hours": hours,
            "delta_t": window_aux_features(hours, visible),
            "division": division,
            "visible": visible,
            "p_index": p_index,
            "anchor_row": anchor_row,
            "target_row": target_row,
            "target_valid": target_valid,
            "target_crosses_division": target_crosses_division,
            "negative_row": negative_row,
            "is_causal": is_causal,
        }


@dataclass(frozen=True)
class EvalWindowSpec:
    """How to build inference windows around each row.

    ``history_budget`` counts frames including the anchor. ``future_budget`` is the
    number of frames after the anchor that may be visible (0 = causal). Frames at
    offsets in ``hide_offsets`` after the anchor are always hidden, matching the
    training distribution where prediction targets are never in context.
    ``no_lineage`` truncates the history at the anchor's own track start.
    ``shuffle`` permutes the embedding content among visible non-anchor frames
    while keeping their timestamps, destroying temporal order.
    """

    history_budget: int
    future_budget: int = 0
    hide_offsets: tuple[int, ...] = ()
    no_lineage: bool = False
    shuffle: bool = False


def build_eval_windows(
    table: LineageTable,
    rows: np.ndarray,
    spec: EvalWindowSpec,
    *,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Build padded inference windows for ``rows`` on their canonical paths.

    Returns row indices (``window_rows``, ``-1`` for padding) rather than gathered
    embeddings so the gather can happen on the training device.
    """
    rows = np.asarray(rows, dtype=int)
    if spec.shuffle and rng is None:
        raise ValueError("shuffle requires an rng")
    L_max = spec.history_budget + spec.future_budget
    B = len(rows)
    window_rows = np.full((B, L_max), -1, dtype=np.int64)
    hours = np.zeros((B, L_max), dtype=np.float32)
    division = np.zeros((B, L_max), dtype=np.float32)
    visible = np.zeros((B, L_max), dtype=bool)
    p_index = np.zeros(B, dtype=np.int64)
    for b, row in enumerate(rows):
        path, p = table.path_of(int(row))
        start = max(0, p - spec.history_budget + 1)
        if spec.no_lineage:
            own = np.flatnonzero(table.track_id[path] == table.track_id[row])
            start = max(start, int(own[0]))
        end = min(len(path) - 1, p + spec.future_budget)
        window = path[start : end + 1].copy()
        L = len(window)
        vis = np.ones(L, dtype=bool)
        if spec.hide_offsets:
            offsets = table.t[window] - table.t[row]
            vis &= ~np.isin(offsets, spec.hide_offsets)
        vis[p - start] = True
        content = window.copy()
        if spec.shuffle:
            movable = np.flatnonzero(vis)
            movable = movable[movable != p - start]
            if len(movable) > 1:
                content[movable] = content[rng.permutation(movable)]
        window_rows[b, :L] = content
        hours[b, :L] = table.hours[window]
        hours[b, L:] = table.hours[row]
        division[b, :L] = table.division_boundary[content]
        visible[b, :L] = vis
        p_index[b] = p - start
    return {
        "window_rows": window_rows,
        "hours": hours,
        "delta_t": window_aux_features(hours, visible),
        "division": division,
        "visible": visible,
        "p_index": p_index,
        "anchor_row": rows,
    }


def smooth_along_paths(
    table: LineageTable,
    features: np.ndarray,
    sigma_hours: float,
    *,
    causal: bool,
) -> np.ndarray:
    """Gap-aware Gaussian smoothing along each row's canonical lineage path.

    With ``causal=True`` only frames at or before each row contribute, which is
    the variant admissible for prospective (pre-onset) probes.
    """
    output = np.empty_like(features, dtype=np.float32)
    done = np.zeros(len(features), dtype=bool)
    for path in table.paths:
        rows = path[~done[path]]
        if not len(rows):
            continue
        h = table.hours[path]
        delta = h[:, None] - h[None, :]
        weights = np.exp(-0.5 * (delta / sigma_hours) ** 2)
        if causal:
            weights = np.tril(weights)
        weights /= weights.sum(axis=1, keepdims=True)
        smoothed = weights @ features[path]
        select = ~done[path]
        output[path[select]] = smoothed[select]
        done[path[select]] = True
    return output


def causal_history_stack(table: LineageTable, features: np.ndarray, k: int) -> np.ndarray:
    """Last ``k`` frames of each row's lineage history, first frame repeated when short.

    Returns shape ``(rows, k, D)`` ordered oldest to newest.
    """
    output = np.empty((len(features), k, features.shape[1]), dtype=np.float32)
    for row in range(len(features)):
        path, p = table.path_of(row)
        positions = np.clip(np.arange(p - k + 1, p + 1), 0, len(path) - 1)
        output[row] = features[path[positions]]
    return output
