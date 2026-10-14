"""Review exact cytoplasm area coverage of existing April masks on compute.

No inference, thresholds, border clamping, or changes to source artifacts.
Track/frame anchors are deduplicated across reporter rows. Native boxes use
int(center) and half-open bounds; 338 is 2*round(128*.1494/.1133).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import sys
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

ANCHOR_KEYS = ["dataset", "fov", "t", "track_id"]
CELL_KEYS = ["dataset", "fov", "t", "cell_id"]
SIDES = (256, 338, 512)


def bounds(y: float, x: float, side: int) -> tuple[int, int, int, int]:
    """Return an unclamped half-open native square, truncating the center."""
    half = side // 2
    return int(y) - half, int(x) - half, int(y) + half, int(x) + half


def count_inside(labels: np.ndarray, cell_id: int, box: tuple[int, int, int, int]) -> int:
    """Count actual labeled pixels inside the square intersection with the image."""
    y0, x0, y1, x1 = box
    y0, x0 = max(0, y0), max(0, x0)
    y1, x1 = min(labels.shape[0], y1), min(labels.shape[1], x1)
    if y1 <= y0 or x1 <= x0:
        return 0
    return int(np.count_nonzero(labels[y0:y1, x0:x1] == cell_id))


def verify_geometry() -> dict:
    """Check exact area counting, negative bounds, and native truncation."""
    labels = np.zeros((8, 8), dtype=np.uint32)
    labels[1:7, 1:7] = 1
    labels[2:6, 2:6] = 0  # irregular hollow mask: bbox area is not its pixel area
    assert np.count_nonzero(labels == 1) == 20
    assert count_inside(labels, 1, bounds(4.9, 4.9, 4)) == 0
    assert count_inside(labels, 1, bounds(2.9, 2.9, 4)) == 5
    assert count_inside(labels, 1, (-4, -4, 2, 2)) == 1
    assert count_inside(labels, 1, (-8, -8, -1, -1)) == 0
    assert count_inside(labels, 1, bounds(4, 4, 8)) == 20
    assert bounds(5.9, 6.9, 256) == (-123, -122, 133, 134)
    assert 2 * round(128 * 0.1494 / 0.1133) == 338
    return {"passed": True, "checks": 8, "bbox_area_not_used_as_cytoplasm_area": True}


def file_sha(path: Path) -> str:
    """Hash an input file without retaining its contents."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reserve_attempt(parent: Path) -> Path:
    """Inspect existing attempt numbers and reserve the next directory atomically."""
    while True:
        numbers = [
            int(match.group(1))
            for path in parent.iterdir()
            if path.is_dir() and (match := re.match(r"^(\d+)_", path.name))
        ]
        target = parent / f"{max(numbers, default=-1) + 1:02d}_cytoplasm_coverage"
        try:
            target.mkdir(exist_ok=False)
            return target
        except FileExistsError:
            continue


def preview_selection(preview: Path) -> list[tuple[str, int]]:
    """Recover the actual processed preview frames, rather than CLI-free config defaults."""
    selection = []
    for path in sorted((preview / "frames").rglob("status.json")):
        relative = path.relative_to(preview / "frames")
        selection.append(("/".join(relative.parts[:-2]), int(relative.parts[-2][1:])))
    expected_times = {2, 30, 60}
    fovs = sorted({fov for fov, _ in selection})
    wells = {"/".join(fov.split("/")[:2]) for fov in fovs}
    if len(selection) != 18 or len(fovs) != 6 or len(wells) != 6:
        raise ValueError(f"Expected six preview FOVs / wells and 18 frames: {selection}")
    if any({t for f, t in selection if f == fov} != expected_times for fov in fovs):
        raise ValueError(f"Preview times must be exactly t2, t30, t60: {selection}")
    return selection


def load_frame(group, fov: str, t: int) -> dict[str, np.ndarray]:
    """Read the already stored native cells, cytoplasm, and nuclear footprints."""
    position = group[fov]
    return {
        "cells": np.asarray(position["0"][t, 0, 0]),
        "cytoplasm": np.asarray(position["labels/cytoplasm/0"][t, 0, 0]),
        "nuclei": np.asarray(position["labels/nuclei/0"][t, 0, 0]),
    }


def measure_frame(rows: pd.DataFrame, masks: dict[str, np.ndarray], cells: pd.DataFrame) -> pd.DataFrame:
    """Measure exact mask pixels for each deduplicated nuclear anchor."""
    cyto = masks["cytoplasm"]
    counts = np.bincount(cyto.ravel())
    frame_cells = cells.set_index("cell_id", verify_integrity=True)
    records = []
    for row in rows.to_dict("records"):
        matched = row["assignment_status"] == "matched"
        cell_id = int(row["cell_id"]) if pd.notna(row["cell_id"]) else 0
        area = int(counts[cell_id]) if matched and 0 < cell_id < len(counts) else 0
        row["cytoplasm_area_px_exact"] = area if matched else np.nan
        row["coverage_valid"] = bool(matched and area > 0)
        if matched:
            if cell_id not in frame_cells.index or area != int(frame_cells.loc[cell_id, "cytoplasm_area_px"]):
                raise ValueError(f"Stored cytoplasm area disagrees with mask pixels: {row}")
            actual = np.argwhere(masks["cells"] == cell_id)
            if not len(actual):
                raise ValueError(f"Matched cell absent from mask: {row}")
            y0, x0 = actual.min(axis=0)
            y1, x1 = actual.max(axis=0) + 1
            if tuple(int(row[k]) for k in ("bbox_y0", "bbox_x0", "bbox_y1", "bbox_x1")) != (y0, x0, y1, x1):
                raise ValueError("Stored cell bbox disagrees with the native mask")
            row["bbox_center_y"] = (int(y0) + int(y1) - 1) / 2
            row["bbox_center_x"] = (int(x0) + int(x1) - 1) / 2
        for side in SIDES:
            box = bounds(row["y"], row["x"], side)
            row[f"nuclear_box_{side}_intersects_image_border"] = (
                box[0] < 0 or box[1] < 0 or box[2] > cyto.shape[0] or box[3] > cyto.shape[1]
            )
            numerator = count_inside(cyto, cell_id, box) if row["coverage_valid"] else np.nan
            row[f"cytoplasm_pixels_in_nuclear_{side}"] = numerator
            row[f"fraction_nuclear_{side}"] = numerator / area if row["coverage_valid"] else np.nan
        numerator = (
            count_inside(cyto, cell_id, bounds(row["bbox_center_y"], row["bbox_center_x"], 256))
            if row["coverage_valid"]
            else np.nan
        )
        row["cytoplasm_pixels_in_bbox_256"] = numerator
        row["fraction_bbox_256"] = numerator / area if row["coverage_valid"] else np.nan
        row["bbox_minus_nuclear_256"] = row["fraction_bbox_256"] - row["fraction_nuclear_256"]
        row["bbox_width_px"] = row["bbox_x1"] - row["bbox_x0"]
        row["bbox_height_px"] = row["bbox_y1"] - row["bbox_y0"]
        records.append(row)
    return pd.DataFrame(records)


def summarize(population: pd.DataFrame) -> list[dict]:
    """Separate measurable anchors, border status, and assigned-nucleus strata."""
    valid = population[population.coverage_valid]
    groups = []
    for border_name, border_group in (
        ("all_matched", valid),
        ("nonborder_matched", valid[~valid.touches_border.fillna(True)]),
        ("border_matched", valid[valid.touches_border.fillna(True)]),
    ):
        for nucleus_name, group in (
            ("all_nuclei", border_group),
            ("single_assigned_nucleus", border_group[border_group.nuclei_in_cell == 1]),
            ("multiple_assigned_nuclei", border_group[border_group.nuclei_in_cell > 1]),
        ):
            for box_name in ("nuclear_256", "nuclear_338", "nuclear_512", "bbox_256"):
                values = group[f"fraction_{box_name}"]
                groups.append(
                    {
                        "population": border_name,
                        "nucleus_stratum": nucleus_name,
                        "box": box_name,
                        "n_track_frame_anchors": len(group),
                        "n_unique_cell_frame_masks": len(group.drop_duplicates(CELL_KEYS)),
                        "median_cytoplasm_fraction": float(values.median()) if len(values) else None,
                        "n_reaching_80_percent": int((values >= 0.8).sum()),
                        "proportion_reaching_80_percent": float((values >= 0.8).mean()) if len(values) else None,
                    }
                )
    return groups


def example_rows(population: pd.DataFrame, targeted: pd.DataFrame) -> pd.DataFrame:
    """Choose representative single-nucleus quantiles and separately requested outliers."""
    pool = population[
        population.coverage_valid & ~population.touches_border.fillna(True) & (population.nuclei_in_cell == 1)
    ]
    if pool.empty:
        raise ValueError("No single-nucleus, non-border matched examples")
    rows = []
    for label, quantile in (("single low (10th percentile)", 0.1), ("single median", 0.5), ("single high (90th)", 0.9)):
        value = pool.fraction_nuclear_256.quantile(quantile)
        row = pool.loc[(pool.fraction_nuclear_256 - value).abs().idxmin()].copy()
        row["example_role"] = label
        rows.append(row)
    for _, row in targeted.iterrows():
        row = row.copy()
        row["example_role"] = "targeted shared-mask outlier; outside denominator"
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def padded_crop(array: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Show an exact 512-square context, padding outside-FOV pixels without shifting it."""
    y0, x0, y1, x1 = box
    result = np.zeros((y1 - y0, x1 - x0), dtype=array.dtype)
    iy0, ix0 = max(0, y0), max(0, x0)
    iy1, ix1 = min(array.shape[0], y1), min(array.shape[1], x1)
    result[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] = array[iy0:iy1, ix0:ix1]
    return result


def render_examples(target: Path, attempt: Path, config: dict, examples: pd.DataFrame, mask_group) -> list[dict]:
    """Render raw membrane/GFP MIPs plus filled cytoplasm, cell outline, and nuclei."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from skimage.segmentation import find_boundaries

    from dynaclr.phenotyping.io import read_position

    fig, axes = plt.subplots(len(examples), 3, figsize=(16, 5.0 * len(examples)), squeeze=False)
    rendered_inputs = []
    for axes_row, row in zip(axes, examples.itertuples(), strict=True):
        masks = load_frame(mask_group, row.fov, row.t)
        box = bounds(row.y, row.x, 512)
        y0, x0, y1, x1 = box
        status_path = attempt / "frames" / row.fov / f"t{row.t:06d}" / "status.json"
        status = json.loads(status_path.read_text())
        position = read_position(Path(config["source"]) / row.fov)
        assert all(abs(value - 0.1133) < 1e-8 for value in position.pixel_size_yx_um)
        channels = [
            config["membrane_channel"],
            next(ch["name"] for ch in config["fluorescence"] if ch["marker"] == "TOMM20"),
        ]
        iy0, ix0 = max(0, y0), max(0, x0)
        iy1, ix1 = min(position.shape[-2], y1), min(position.shape[-1], x1)
        images = []
        scales = []
        with closing(zarr.storage.LocalStore(str(position.path), read_only=True)) as store:
            source = zarr.open_group(store=store, mode="r")[position.array_path]
            for channel in channels:
                raw = np.asarray(
                    source[
                        row.t, position.channels.index(channel), status["z_start"] : status["z_stop"], iy0:iy1, ix0:ix1
                    ]
                )
                pixels = raw.max(axis=0)
                display = np.zeros((512, 512), dtype=pixels.dtype)
                display[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] = pixels
                low, high = np.percentile(pixels, (1, 99))
                scales.append((float(low), float(high)))
                images.append(np.clip((display - low) / max(float(high - low), 1e-9), 0, 1))
                rendered_inputs.append(
                    {
                        "fov": row.fov,
                        "t": int(row.t),
                        "track_id": int(row.track_id),
                        "channel": channel,
                        "z_start": status["z_start"],
                        "z_stop": status["z_stop"],
                        "context_native_bounds": list(box),
                        "raw_slab_sha256": hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest(),
                        "display_p1_p99": [float(low), float(high)],
                    }
                )
        cell = padded_crop(masks["cells"], box) == row.cell_id
        cyto = padded_crop(masks["cytoplasm"], box) == row.cell_id
        nuclei = padded_crop(masks["nuclei"], box)
        overlay = np.repeat(images[1][..., None], 3, axis=-1)
        overlay[cyto] = 0.65 * overlay[cyto] + 0.35 * np.array([0, 1, 0.7])
        overlay[nuclei > 0] = 0.5 * overlay[nuclei > 0] + 0.5 * np.array([0.2, 0.5, 1])
        overlay[find_boundaries(cell, mode="inner")] = [1, 0, 1]
        overlay[find_boundaries(nuclei == row.nucleus_label, mode="inner")] = [1, 0.65, 0]
        panels = (images[0], images[1], overlay)
        titles = (
            "Raw virtual membrane MIP",
            "Raw GFP / TOMM20 MIP",
            "Filled cytoplasm + cell outline + nucleus footprints",
        )
        for axis, pixels, title in zip(axes_row, panels, titles, strict=True):
            axis.imshow(pixels, cmap="gray", origin="upper")
            axis.add_patch(Rectangle((127.5, 127.5), 256, 256, fill=False, edgecolor="yellow", linewidth=1.8))
            bbox_y, bbox_x = int(row.bbox_center_y) - y0, int(row.bbox_center_x) - x0
            axis.add_patch(
                Rectangle(
                    (bbox_x - 128.5, bbox_y - 128.5),
                    256,
                    256,
                    fill=False,
                    edgecolor="cyan",
                    linewidth=1.2,
                    linestyle="--",
                )
            )
            axis.plot(256, 256, "+", color="yellow", markersize=9)
            axis.set_xlim(-0.5, 511.5)
            axis.set_ylim(511.5, -0.5)
            axis.set_title(title, fontsize=10)
            axis.axis("off")
        label = (
            f"{row.example_role}\n{row.fov} track {row.track_id}, t{row.t}, cell {row.cell_id}; "
            f"assigned nuclei={row.nuclei_in_cell}\n"
            f"bbox {int(row.bbox_width_px)}×{int(row.bbox_height_px)} native px; "
            f"cytoplasm {int(row.cytoplasm_area_px_exact):,} px\n"
            f"exact area fraction: nuclear 256={row.fraction_nuclear_256:.1%}, "
            f"bbox 256={row.fraction_bbox_256:.1%}; nuclear 338={row.fraction_nuclear_338:.1%}"
        )
        axes_row[0].text(0, -0.03, label, transform=axes_row[0].transAxes, fontsize=9, va="top")
        for axis, (low, high) in zip(axes_row[:2], scales, strict=True):
            axis.text(
                0,
                0.01,
                f"display p1–p99: {low:.1f}–{high:.1f} raw units",
                transform=axis.transAxes,
                color="white",
                fontsize=8,
                bbox={"facecolor": "black", "alpha": 0.6},
            )
    fig.suptitle(
        "Exact native cytoplasm area review — every panel is a 512×512 native context\n"
        "Yellow: nuclear-centered 256 square; cyan dashed: bbox-centered 256. "
        "Teal: cytoplasm; magenta: cell; blue: nuclei; orange: tracked nucleus.\n"
        "Five-plane stored-status MIPs; display stretch only. Unclamped boxes; outside-FOV context is padded. "
        "Not a resolved model-crop audit.",
        fontsize=12,
    )
    fig.subplots_adjust(top=0.95, bottom=0.045, hspace=0.48, wspace=0.06)
    fig.savefig(target / "coverage_examples.png", dpi=150)
    fig.savefig(target / "coverage_examples.pdf")
    plt.close(fig)
    return rendered_inputs


def review(attempt: Path, preview: Path, output: Path | None = None) -> dict:
    """Create one new numbered review from existing masks and bounded source crops."""
    from dynaclr.phenotyping.io import assert_compute_node

    assert_compute_node()
    checks = verify_geometry()
    selection = preview_selection(preview)
    manifest = json.loads((attempt / "manifest.json").read_text())
    config = manifest["scientific"]["config"]
    observations = pd.read_parquet(attempt / "observations.parquet")
    cells = pd.read_parquet(attempt / "cells.parquet")
    selected_keys = set(selection)
    targeted_keys = {("A/2/0001", 38, 94), ("A/2/0001", 24, 93)}
    fields = ANCHOR_KEYS + [
        "y",
        "x",
        "nucleus_label",
        "cell_id",
        "assignment_status",
        "nuclei_in_cell",
        "touches_border",
        "bbox_y0",
        "bbox_x0",
        "bbox_y1",
        "bbox_x1",
    ]
    selected = observations[[((f, int(t)) in selected_keys) for f, t in zip(observations.fov, observations.t)]]
    targeted = observations[
        [
            (f, int(t), int(track)) in targeted_keys
            for f, t, track in zip(observations.fov, observations.t, observations.track_id)
        ]
    ]
    anchors = pd.concat([selected[fields], targeted[fields]]).drop_duplicates()
    if anchors.duplicated(ANCHOR_KEYS).any():
        raise ValueError("Reporter copies disagree about track/frame geometry or assignment")
    anchors["population_frame"] = [(f, int(t)) in selected_keys for f, t in zip(anchors.fov, anchors.t)]
    if output is None:
        target = reserve_attempt(attempt.parent)
    else:
        target = output.resolve()
        if target.parent != attempt.resolve().parent or (target / "summary.json").exists():
            raise ValueError("Only an unfinished owned review attempt may be reused")
        prior = json.loads((target / "provenance.json").read_text())
        if prior["source_attempt"] != str(attempt.resolve()) or "cytoplasm_coverage" not in target.name:
            raise ValueError("Output is not this source's unfinished coverage review")
    print(f"RESERVED {target}", flush=True)
    provenance_dir = target / "provenance"
    provenance_dir.mkdir(exist_ok=output is not None)
    script = Path(__file__).resolve()
    shutil.copy2(script, provenance_dir / script.name)
    for name in ("manifest.json", "pilot.yml", "summary.json"):
        shutil.copy2(attempt / name, provenance_dir / f"source_{name}")
    input_files = {
        str(attempt / name): file_sha(attempt / name)
        for name in ("manifest.json", "pilot.yml", "summary.json", "observations.parquet", "cells.parquet")
    }
    provenance = {
        "purpose": "bounded exact cytoplasm area coverage review; no inference or active cutoff",
        "source_attempt": str(attempt.resolve()),
        "preview_attempt": str(preview.resolve()),
        "output_attempt": str(target.resolve()),
        "script_sha256": file_sha(script),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": sys.executable,
        "argv": sys.argv,
        "input_file_sha256": input_files,
        "source_frozen_code": manifest["code"],
        "source_config": config,
        "versions": {
            name: importlib.metadata.version(name) for name in ("numpy", "pandas", "zarr", "matplotlib", "scikit-image")
        },
        "selection": [{"fov": f, "t": t} for f, t in selection],
        "native_pixel_size_um": 0.1133,
        "reference_pixel_size_um": 0.1494,
        "sides_native_px": list(SIDES),
        "square_rule": "int(center), [center-half:center+half]; no border clamp",
        "bbox_center_rule": "int((bbox_min + bbox_max_exclusive - 1)/2)",
        "geometry_checks": checks,
    }
    (target / "provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    results, mask_inputs = [], []
    with closing(zarr.storage.LocalStore(str(attempt / "segmentations.ome.zarr"), read_only=True)) as store:
        group = zarr.open_group(store=store, mode="r")
        for (fov, t), rows in anchors.groupby(["fov", "t"], sort=True):
            status_path = attempt / "frames" / fov / f"t{t:06d}" / "status.json"
            status = json.loads(status_path.read_text())
            if status["state"] != "complete":
                raise ValueError(f"Selected source frame is not complete: {status_path}")
            masks = load_frame(group, fov, int(t))
            assert np.array_equal(masks["cytoplasm"], np.where(masks["nuclei"] == 0, masks["cells"], 0))
            frame_cells = cells[(cells.fov == fov) & (cells.t == t)]
            measured = measure_frame(rows, masks, frame_cells)
            results.append(measured)
            mask_inputs.append(
                {
                    "fov": fov,
                    "t": int(t),
                    "status_sha256": file_sha(status_path),
                    "shape_yx": list(masks["cells"].shape),
                    "mask_array_sha256": {
                        k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in masks.items()
                    },
                }
            )
            print(f"MEASURED {fov} t{t}: {len(rows)} anchors", flush=True)
        coverage = pd.concat(results, ignore_index=True)
        coverage["touches_border"] = coverage["touches_border"].fillna(True).astype(bool)
        population = coverage[coverage.population_frame].copy()
        targeted = coverage[~coverage.population_frame].copy()
        if set(zip(targeted.fov, targeted.t, targeted.track_id)) != targeted_keys:
            raise ValueError("Both requested targeted shared-mask examples must be present")
        examples = example_rows(population, targeted)
        raw_inputs = render_examples(target, attempt, config, examples, group)
    coverage.to_csv(target / "anchor_coverage.csv", index=False)
    examples.to_csv(target / "example_observations.csv", index=False)
    statistics = summarize(population)
    pd.DataFrame(statistics).to_csv(target / "coverage_summary.csv", index=False)
    sizes = population[population.coverage_valid].groupby(CELL_KEYS).size()
    primary = population[population.coverage_valid & ~population.touches_border.fillna(True)]
    primary_sizes = primary.groupby(CELL_KEYS).size()
    counts = {
        "selected_frames": len(selection),
        "selected_wells": 6,
        "reporter_rows_before_deduplication": len(selected),
        "deduplicated_track_frame_anchors": len(population),
        "assignment_status_counts": population.assignment_status.value_counts().to_dict(),
        "valid_area_anchors": int(population.coverage_valid.sum()),
        "unique_matched_cell_frame_masks": len(sizes),
        "shared_cell_frame_masks": int((sizes > 1).sum()),
        "anchors_sharing_cell_frame_masks": int(sizes[sizes > 1].sum()),
        "extra_anchors_over_unique_masks": int((sizes - 1).sum()),
        "primary_nonborder_matched_anchors": len(primary),
        "primary_unique_cell_frame_masks": len(primary_sizes),
        "primary_shared_cell_frame_masks": int((primary_sizes > 1).sum()),
        "primary_anchors_sharing_masks": int(primary_sizes[primary_sizes > 1].sum()),
        "targeted_examples_outside_population": len(targeted),
    }
    report = {
        "complete": True,
        "counts": counts,
        "statistics": statistics,
        "native_sides_um": {str(side): side * 0.1133 for side in SIDES},
        "primary_bbox_minus_nuclear_256_median": float(primary.bbox_minus_nuclear_256.median()),
        "interpretation": "Exact cytoplasm mask pixel fractions, not complete-cell bbox containment. "
        "Track/frame anchors can share masks and repeat cells over time; not independent cells. "
        "Targeted merged-mask examples are excluded from all population statistics. "
        "Unclamped native squares do not certify actual model crops/slabs or segmentation accuracy.",
        "active_filter": None,
        "provenance": "provenance.json",
        "script_sha256": provenance["script_sha256"],
        "geometry_checks": checks,
    }
    (target / "mask_inputs.json").write_text(json.dumps(mask_inputs, indent=2, allow_nan=False) + "\n")
    (target / "rendered_raw_inputs.json").write_text(json.dumps(raw_inputs, indent=2, allow_nan=False) + "\n")
    # Source files are hashed again before declaring completion; no completed input is mutated.
    if any(file_sha(Path(path)) != digest for path, digest in input_files.items()):
        raise RuntimeError("Source tables/config changed during the review")
    (target / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(target), "counts": counts, "statistics": statistics}, indent=2), flush=True)
    return report


def main() -> None:
    """Run the bounded review or only its synthetic geometry checks on compute."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-attempt", type=Path)
    parser.add_argument("--preview-attempt", type=Path)
    parser.add_argument(
        "--output", type=Path, help="Reuse only this review's unfinished owned attempt after a failure."
    )
    parser.add_argument("--verify-geometry", action="store_true")
    args = parser.parse_args()
    if args.verify_geometry:
        from dynaclr.phenotyping.io import assert_compute_node

        assert_compute_node()
        print(json.dumps(verify_geometry()))
        return
    if args.source_attempt is None:
        parser.error("--source-attempt is required")
    preview = args.preview_attempt or args.source_attempt.parent / "02_grayscale_058_pilot"
    review(args.source_attempt, preview, args.output)


if __name__ == "__main__":
    main()
