"""Pure algorithms: projection, verified tracking joins and multinuclear cells."""

from __future__ import annotations

import numpy as np
import pandas as pd


class FrameSkip(ValueError):
    """A known missing/invalid observation, not an inference/programming error."""


def projection_window(focus: object, depth: int) -> slice:
    """Return the exact five-plane window or a reason to skip the frame."""
    if isinstance(focus, (bool, np.bool_)) or focus is None:
        raise FrameSkip("missing_or_invalid_focus")
    try:
        value = float(focus)
    except (TypeError, ValueError):
        raise FrameSkip("missing_or_invalid_focus") from None
    if not np.isfinite(value) or value != int(value):
        raise FrameSkip("missing_or_invalid_focus")
    start, stop = int(value) - 1, int(value) + 4
    if start < 0 or stop > depth:
        raise FrameSkip("focus_window_out_of_bounds")
    return slice(start, stop)


def project(stack: np.ndarray) -> np.ndarray:
    """Maximum-project five finite Z planes without changing raw intensity units."""
    if stack.ndim != 3 or stack.shape[0] != 5:
        raise FrameSkip("projection_requires_five_planes")
    if not np.isfinite(stack).all():
        raise FrameSkip("nonfinite_projection_input")
    return np.max(stack, axis=0)


def verify_phase_alignment(assembled: np.ndarray, prepared: np.ndarray, rtol: float = 0, atol: float = 0) -> None:
    """Pixel proof for legacy prepared metadata whose time scale is placeholder 1."""
    if assembled.shape != prepared.shape or not np.isfinite(assembled).all() or not np.isfinite(prepared).all():
        raise FrameSkip("phase_coordinate_alignment_failed")
    if not np.allclose(assembled, prepared, rtol=rtol, atol=atol):
        raise FrameSkip("phase_coordinate_alignment_failed")
    if np.ptp(assembled) == 0:
        raise FrameSkip("phase_alignment_uninformative")


def validate_tracks(tracks: pd.DataFrame) -> pd.DataFrame:
    """Validate integral observation identities and unique track-time rows."""
    required = {"t", "track_id", "id", "y", "x"}
    if required - set(tracks):
        raise ValueError(f"Track CSV missing {sorted(required - set(tracks))}")
    result = tracks.copy()
    for col in ("t", "track_id", "id"):
        values = pd.to_numeric(result[col], errors="raise")
        if not np.isfinite(values).all() or (values != np.floor(values)).any():
            raise ValueError(f"Noninteger tracking column {col}")
        result[col] = values.astype(np.int64)
    if (result.t < 0).any() or (result.track_id <= 0).any() or (result.id <= 0).any():
        raise ValueError("Invalid time or tracking identifier")
    if result.duplicated(["t", "track_id"]).any() or result.duplicated(["t", "id"]).any():
        raise ValueError("Duplicate tracking observations")
    if not np.isfinite(result[["y", "x"]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite tracking coordinates")
    return result


def resolve_track_labels(
    nuclei: np.ndarray, tracks: pd.DataFrame, mode: str = "auto", stride: int = 1_000_000
) -> pd.DataFrame:
    """Choose label encoding with pixel evidence, never just matching integer sets.

    Require every existing expected footprint to contain its rounded CSV centroid.
    Absent labels are retained as missing observations. Multiple encodings that
    produce the exact same row labels are equivalent; differing valid encodings
    are rejected. A concave footprint whose centroid falls outside is skipped for
    review rather than silently assigned to a nearby object.
    """
    result = tracks.copy()
    if result.empty:
        if np.any(nuclei):
            raise FrameSkip("nuclear_labels_without_track_rows")
        result["nucleus_label"] = pd.Series(dtype="int64")
        result["label_encoding"] = pd.Series(dtype="str")
        result["nucleus_status"] = pd.Series(dtype="str")
        return result
    present = np.unique(nuclei)
    present = present[present > 0]
    ys = np.rint(result.y).astype(np.int64).to_numpy()
    xs = np.rint(result.x).astype(np.int64).to_numpy()
    if ((ys < 0) | (xs < 0) | (ys >= nuclei.shape[0]) | (xs >= nuclei.shape[1])).any():
        raise FrameSkip("track_centroid_out_of_bounds")
    observed = nuclei[ys, xs]
    candidates = {
        key: result[key].to_numpy(dtype=np.int64)
        for key in ("track_id", "id", "label_id", "segmentation_id")
        if key in result
    }
    # Only try the encoded detection IDs when the time prefix is verified.
    if np.all(result.id.to_numpy() // stride == result.t.to_numpy() + 1):
        candidates["id_modulo"] = result.id.to_numpy() % stride
    if mode != "auto":
        candidates = {key: value for key, value in candidates.items() if key == mode}
    valid: list[tuple[str, np.ndarray]] = []
    for name, labels in candidates.items():
        exists = np.isin(labels, present)
        if np.any(labels <= 0) or len(np.unique(labels)) != len(labels):
            continue
        if not set(present).issubset(set(labels)):
            continue  # must account for every nuclear footprint
        if len(present) and np.all(observed[exists] == labels[exists]) and np.all(observed[~exists] == 0):
            valid.append((name, labels))
    if not len(present):
        result["nucleus_label"] = pd.array([pd.NA] * len(result), dtype="Int64")
        result["label_encoding"] = "unverified_empty_mask"
        result["nucleus_status"] = "missing_nuclear_footprint"
        return result
    if not valid:
        raise FrameSkip("track_label_encoding_unverified")
    if any(not np.array_equal(valid[0][1], labels) for _, labels in valid[1:]):
        raise FrameSkip("track_label_encoding_ambiguous")
    name, labels = valid[0]
    result["nucleus_label"] = labels
    result["label_encoding"] = name
    result["nucleus_status"] = np.where(np.isin(labels, present), "verified", "missing_nuclear_footprint")
    return result


def map_nuclei_to_cells(cells: np.ndarray, nuclei: np.ndarray, tracks: pd.DataFrame) -> pd.DataFrame:
    """Assign each verified nuclear footprint to a cell with strict majority overlap."""
    if cells.shape != nuclei.shape:
        raise FrameSkip("nuclear_cell_shape_mismatch")
    records = []
    for row in tracks.to_dict("records"):
        label = row.get("nucleus_label")
        record = dict(
            row,
            cell_id=pd.NA,
            overlap_fraction=np.nan,
            nucleus_area_px=0,
            assignment_status="missing_nuclear_footprint",
        )
        if label is not None and not pd.isna(label):
            footprint = nuclei == int(label)
            area = int(footprint.sum())
            record["nucleus_area_px"] = area
            if area:
                ids, counts = np.unique(cells[footprint], return_counts=True)
                fg = ids > 0
                ids, counts = ids[fg], counts[fg]
                if not len(ids):
                    record["assignment_status"] = "unmatched"
                    record["overlap_fraction"] = 0.0
                else:
                    winner = int(np.argmax(counts))
                    fraction = float(counts[winner] / area)
                    record["overlap_fraction"] = fraction
                    record["assignment_status"] = "matched" if fraction > 0.5 else "ambiguous"
                    if fraction > 0.5:
                        record["cell_id"] = int(ids[winner])
        records.append(record)
    result = pd.DataFrame.from_records(records)
    if result.empty:
        result = tracks.copy()
        for name, dtype in [
            ("cell_id", "Int64"),
            ("overlap_fraction", "float64"),
            ("nucleus_area_px", "int64"),
            ("assignment_status", "str"),
        ]:
            result[name] = pd.Series(dtype=dtype)
    result["cell_id"] = pd.array(result["cell_id"], dtype="Int64")
    counts = result.loc[result.assignment_status == "matched", "cell_id"].value_counts()
    result["nuclei_in_cell"] = result.cell_id.map(counts).fillna(0).astype(int)
    return result


def cytoplasm_labels(cells: np.ndarray, nuclei: np.ndarray) -> np.ndarray:
    """Remove ALL footprints, including unmatched nuclei inside another cell."""
    if cells.shape != nuclei.shape:
        raise ValueError("Cell and nucleus masks must have matching XY")
    result = cells.copy()
    result[nuclei > 0] = 0
    return result


def summarize_tracks(observations: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    """Coverage against full CSV trajectory, with unknown signal kept nullable.

    Input observation schema includes dataset/fov/track_id/t/marker, valid_signal,
    and nullable below_threshold. Missing frames break contiguous low intervals.
    """
    keys = ["dataset", "fov", "track_id"]
    if observations.empty:
        return pd.DataFrame(
            columns=keys
            + [
                "marker",
                "expected_frames",
                "valid_frames",
                "coverage",
                "below_fraction",
                "mostly_low_signal",
                "longest_low_interval_frames",
            ]
        )
    if observations.duplicated(keys + ["t", "marker"]).any():
        raise ValueError("Duplicate per-track fluorescence observations")
    records = []
    for group_key, group in observations.groupby(keys + ["marker"], dropna=False):
        full = expected
        for key, value in zip(keys, group_key[:3]):
            full = full[full[key] == value]
        times = np.sort(full.t.unique())
        valid = group[group.valid_signal.fillna(False)]
        known = valid[valid.below_threshold.notna()]
        fraction = float(known.below_threshold.astype(bool).mean()) if len(known) else np.nan
        # A partially evaluated threshold is not a complete track verdict.
        fraction = fraction if len(known) == len(valid) and len(valid) else np.nan
        longest = current = 0
        lows = set(known.loc[known.below_threshold.astype(bool), "t"].astype(int))
        previous = None
        for time in times:
            current = current + 1 if time in lows and previous == time - 1 else int(time in lows)
            longest = max(longest, current)
            previous = time
        records.append(
            dict(
                zip(keys + ["marker"], group_key),
                expected_frames=len(times),
                observed_frames=len(group),
                valid_frames=len(valid),
                coverage=len(valid) / len(times) if len(times) else np.nan,
                below_fraction=fraction,
                mostly_low_signal=(fraction > 0.5 if np.isfinite(fraction) else None),
                longest_low_interval_frames=longest if len(known) else None,
            )
        )
    result = pd.DataFrame(records)
    result["mostly_low_signal"] = pd.array(result.mostly_low_signal, dtype="boolean")
    return result
