"""CellProfiler measurements with explicit stable-label and compartment joins."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dynaclr.phenotyping.config import Channel


def dense_labels(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Relabel sparse positive IDs for measurement and retain their original identities."""
    ids = np.unique(labels)
    ids = ids[ids > 0]
    dense = np.zeros(labels.shape, dtype=np.int32)
    foreground = labels > 0
    dense[foreground] = np.searchsorted(ids, labels[foreground]) + 1
    return dense, ids


def cell_centers(cells: np.ndarray, cytoplasm: np.ndarray) -> pd.DataFrame:
    """Return bbox midpoints and unweighted cytoplasm centroids in native pixel coordinates.

    Cytoplasm labels must already exclude all nuclear footprints. Each remaining
    mask pixel has equal weight, independent of fluorescence intensity.
    Original sparse cell IDs are retained. Exclusive bbox maxima contribute
    ``maximum - 1`` to pixel-center midpoints. Empty cytoplasm has NaN centers.
    """
    from skimage.measure import regionprops

    if cells.ndim != 2 or cytoplasm.shape != cells.shape:
        raise ValueError("Cell and cytoplasm labels must share a native 2D grid")
    if np.any((cytoplasm != 0) & (cytoplasm != cells)):
        raise ValueError("Cytoplasm must retain its owning cell label")
    dense, ids = dense_labels(cells)
    props = regionprops(dense)
    centers = pd.DataFrame({"cell_id": ids.astype(np.int64)})
    for axis, name in enumerate(("bbox_center_y", "bbox_center_x")):
        centers[name] = np.asarray([(p.bbox[axis] + p.bbox[axis + 2] - 1) / 2 for p in props], dtype=float)
    cyto_dense, cyto_ids = dense_labels(cytoplasm)
    cyto_props = regionprops(cyto_dense)
    for axis, name in enumerate(("cytoplasm_centroid_y", "cytoplasm_centroid_x")):
        centers[name] = centers.cell_id.map({label: p.centroid[axis] for label, p in zip(cyto_ids, cyto_props)})
    return centers


def _features(frame: pd.DataFrame, features: dict, prefix: str) -> pd.DataFrame:
    """Flatten any vector feature without relying on dictionary insertion order."""
    for name, values in features.items():
        values = np.asarray(values)
        if values.ndim == 0 or values.shape[0] != len(frame):
            raise ValueError(f"cp_measure {name} returned inconsistent object count")
        if values.ndim == 1:
            frame[f"{prefix}{name}"] = values
        else:
            for index, column in enumerate(values.reshape(len(frame), -1).T):
                frame[f"{prefix}{name}_{index}"] = column
    return frame


def measure_cells(
    cells: np.ndarray,
    cytoplasm: np.ndarray,
    nuclei: np.ndarray,
    virtual_nucleus: np.ndarray,
    fluorescence: dict[str, np.ndarray],
    channels: tuple[Channel, ...],
    membership: pd.DataFrame,
    pixel_size_yx_um: tuple[float, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure shape and compartment fluorescence once per whole cell."""
    from cp_measure.core.measureobjectintensity import get_intensity
    from cp_measure.core.measureobjectsizeshape import get_sizeshape
    from scipy.ndimage import find_objects
    from skimage.measure import regionprops

    dense, ids = dense_labels(cells)
    geometry = cell_centers(cells, cytoplasm)
    if not len(ids):
        return geometry, pd.DataFrame(columns=["cell_id", "channel_name", "marker", "role", "compartment"])
    with np.errstate(divide="ignore", invalid="ignore"):
        geometry = _features(geometry, get_sizeshape(dense, None, calculate_advanced=False), "cp_")
    props = regionprops(dense)
    py, px = pixel_size_yx_um
    geometry["area_px"] = [p.area for p in props]
    geometry["area_um2"] = geometry.area_px * py * px
    geometry[["bbox_y0", "bbox_x0", "bbox_y1", "bbox_x1"]] = [p.bbox for p in props]
    geometry["bbox_height_px"] = geometry.bbox_y1 - geometry.bbox_y0
    geometry["bbox_width_px"] = geometry.bbox_x1 - geometry.bbox_x0
    geometry["bbox_height_um"] = geometry.bbox_height_px * py
    geometry["bbox_width_um"] = geometry.bbox_width_px * px
    geometry["touches_border"] = (
        (geometry.bbox_y0 == 0)
        | (geometry.bbox_x0 == 0)
        | (geometry.bbox_y1 == cells.shape[0])
        | (geometry.bbox_x1 == cells.shape[1])
    )
    counts = membership.loc[membership.assignment_status == "matched", "cell_id"].value_counts()
    geometry["nuclei_in_cell"] = geometry.cell_id.map(counts).fillna(0).astype(int)
    geometry["multinucleated"] = geometry.nuclei_in_cell > 1
    # Remove every nuclear footprint, even one with ambiguous membership.
    nuclear_inside = np.where(nuclei > 0, cells, 0)
    compartments = {"whole_cell": cells, "cytoplasm": cytoplasm, "nucleus": nuclear_inside}
    frames = []
    for compartment, labels in compartments.items():
        masks, present = dense_labels(labels)
        geometry[f"{compartment}_area_px"] = (
            geometry.cell_id.map(dict(zip(*np.unique(labels[labels > 0], return_counts=True)))).fillna(0).astype(int)
        )
        slices = find_objects(masks)
        for ch in channels:
            if ch.name not in fluorescence:
                raise ValueError(f"Missing fluorescence channel {ch.name}")
            pixels = fluorescence[ch.name]
            if pixels.shape != cells.shape:
                raise ValueError("Fluorescence and masks have different shapes")
            frame = pd.DataFrame({"cell_id": present.astype(np.int64)})
            if len(present):
                frame = _features(frame, get_intensity(masks, pixels.astype(np.float64), legacy=False), "cp_")
                frame["mean_intensity"] = frame["cp_Intensity_MeanIntensity"]
                frame["median_intensity"] = frame["cp_Intensity_MedianIntensity"]
                rois = [pixels[sl][masks[sl] == index + 1] for index, sl in enumerate(slices)]
                frame["p90_intensity"] = [np.percentile(values, 90) for values in rois]
                frame["p99_intensity"] = [np.percentile(values, 99) for values in rois]
                frame["mad_intensity"] = [np.median(np.abs(values - np.median(values))) for values in rois]
                frame["area_px"] = [len(values) for values in rois]
            frame["channel_name"], frame["marker"], frame["role"] = ch.name, ch.marker, ch.role
            frame["compartment"] = compartment
            frames.append(frame)
    # Continuous QC of nuclear-like signal left in cytoplasm; no automatic cutoff.
    cyto_dense, cyto_ids = dense_labels(cytoplasm)
    if len(cyto_ids):
        qc = get_intensity(cyto_dense, virtual_nucleus.astype(np.float64), edge_measurements=False)
        geometry["cytoplasm_virtual_nucleus_mean"] = geometry.cell_id.map(
            dict(zip(cyto_ids, qc["Intensity_MeanIntensity"]))
        )
        slices = find_objects(cyto_dense)
        geometry["cytoplasm_virtual_nucleus_p99"] = geometry.cell_id.map(
            {
                label: float(np.percentile(virtual_nucleus[sl][cyto_dense[sl] == index + 1], 99))
                for index, (label, sl) in enumerate(zip(cyto_ids, slices))
            }
        )
    else:
        geometry["cytoplasm_virtual_nucleus_mean"] = np.nan
        geometry["cytoplasm_virtual_nucleus_p99"] = np.nan
    return geometry, pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def observation_measurements(
    membership: pd.DataFrame, geometry: pd.DataFrame, intensity: pd.DataFrame, channels: tuple[Channel, ...]
) -> pd.DataFrame:
    """One row per nuclear-track/frame/channel; shared cells remain shared."""
    center_columns = ["bbox_center_y", "bbox_center_x", "cytoplasm_centroid_y", "cytoplasm_centroid_x"]
    frames = []
    for ch in channels:
        group = membership.copy()
        group["nucleus_center_y"], group["nucleus_center_x"] = group.y, group.x
        group["channel_name"], group["marker"], group["role"] = ch.name, ch.marker, ch.role
        if len(geometry):
            group = group.merge(
                geometry[
                    [
                        "cell_id",
                        "bbox_y0",
                        "bbox_x0",
                        "bbox_y1",
                        "bbox_x1",
                        "area_px",
                        "area_um2",
                        "touches_border",
                        *center_columns,
                    ]
                ],
                on="cell_id",
                how="left",
                validate="many_to_one",
            )
            distances = np.stack(
                [
                    group.y - group.bbox_y0,
                    group.bbox_y1 - 1 - group.y,
                    group.x - group.bbox_x0,
                    group.bbox_x1 - 1 - group.x,
                ],
                axis=1,
            )
            group["nuclear_centered_patch_side_px"] = 2 * np.ceil(np.max(distances, axis=1)) + 1
        else:
            for name in ("area_px", "area_um2", "nuclear_centered_patch_side_px", *center_columns):
                group[name] = np.nan
        group.loc[group.assignment_status != "matched", center_columns] = np.nan
        if len(intensity):
            selected = intensity[(intensity.channel_name == ch.name) & (intensity.compartment == "cytoplasm")]
            cols = [
                name
                for name in (
                    "cell_id",
                    "mean_intensity",
                    "median_intensity",
                    "p90_intensity",
                    "p99_intensity",
                    "mad_intensity",
                )
                if name in selected
            ]
            group = group.merge(selected[cols], on="cell_id", how="left", validate="many_to_one")
        if "mean_intensity" not in group:
            group["mean_intensity"] = np.nan
        group["valid_signal"] = (group.assignment_status == "matched") & np.isfinite(group.mean_intensity)
        group["below_threshold"] = pd.array([pd.NA] * len(group), dtype="boolean")
        group["eligible"] = pd.array([pd.NA] * len(group), dtype="boolean")
        group["reason"] = np.where(
            group.valid_signal, "no_threshold", np.where(group.assignment_status == "ambiguous", "ambiguous", "missing")
        )
        frames.append(group)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
