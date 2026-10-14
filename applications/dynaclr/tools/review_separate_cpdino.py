"""Bounded separate CP-DINO nuclear/membrane diagnostic; production remains immutable."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import platform
import shutil
import socket
import subprocess
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import zarr
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
from skimage.transform import resize

from dynaclr.phenotyping.config import Segmentation
from dynaclr.phenotyping.io import assert_compute_node, read_position
from dynaclr.phenotyping.segmentation import CPDinoSegmenter, sha256_file

matplotlib.use("Agg")

BASE = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics")
JULY = BASE / "2026_07_01_A549_SEC61B_TOMM20_G3BP1_ZIKV/2-phenotyping/segmentations_membrane"
APRIL = BASE / "2026_04_10_A549_TOMM20_ZIKV/2-phenotyping/segmentations_membrane/03_full_pilot"
REFERENCE = "origin/dynacell-models"
GFP = "raw GFP EX488 EM525-45"


def dump(path: Path, data: dict) -> None:
    """Write a finite JSON diagnostic artifact."""
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def stock_inference(segmenter, image, spacing, target):
    """Preserve working labels and restore their original instance IDs to native XY."""
    shape = (
        image.shape if target is None else tuple(max(32, round(n * p / target)) for n, p in zip(image.shape, spacing))
    )
    working = image.astype(np.float32, copy=False)
    if shape != image.shape:
        working = resize(working, shape, order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
    params = {
        "channel_axis": None,
        "do_3D": False,
        "normalize": True,
        "diameter": None,
        "batch_size": 8,
        "flow_threshold": 0.4,
        "cellprob_threshold": 0.0,
        "min_size": 15,
    }
    started = time.monotonic()
    labels, _, _ = segmenter.model.eval(working, **params)
    labels = np.asarray(labels, dtype=np.uint32)
    assert labels.shape == shape
    native = (
        labels
        if shape == image.shape
        else resize(labels, image.shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint32)
    )
    return native, labels, {"seconds": time.monotonic() - started, "working_shape": list(shape), "kwargs": params}


def overlap_instances(footprint, instances, y, x):
    """Report every overlapping instance, full geometry and original centroid proximity."""
    area = int(footprint.sum())
    ids, counts = np.unique(instances[footprint], return_counts=True)
    props = {r.label: r for r in regionprops(instances)}
    result = []
    for label, count in zip(ids, counts):
        if label == 0:
            continue
        p = props[int(label)]
        cy, cx = p.centroid
        result.append(
            {
                "instance_id": int(label),
                "intersection_px": int(count),
                "fraction_of_tracked_nucleus": int(count) / area,
                "fraction_of_inferred_instance": int(count) / int(p.area),
                "instance_area_px": int(p.area),
                "centroid_y": float(cy),
                "centroid_x": float(cx),
                "centroid_distance_to_track_px": float(np.hypot(cy - y, cx - x)),
                "contains_track_centroid": int(instances[round(y), round(x)]) == int(label),
                "bbox": list(map(int, p.bbox)),
            }
        )
    return sorted(result, key=lambda r: r["intersection_px"], reverse=True)


def intensity_stats(image, footprint):
    """Measure unnormalized GFP without applying an eligibility threshold."""
    values = image[footprint]
    if not len(values):
        return {"area_px": 0, "mean": None, "median": None, "p90": None}
    return {
        "area_px": len(values),
        "mean": float(values.mean(dtype=np.float64)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
    }


def edge(ax, footprint, color):
    """Draw an explicit-color binary boundary."""
    boundary = find_boundaries(footprint)
    ax.imshow(
        np.ma.masked_where(~boundary, np.ones(boundary.shape)), cmap=ListedColormap([color]), vmin=0, vmax=1, alpha=0.9
    )


def gray(ax, image, limits=None):
    """Display a grayscale panel with explicit percentile stretch."""
    low, high = np.nanpercentile(image, [1, 99.5]) if limits is None else limits
    ax.imshow(image, cmap="gray", vmin=low, vmax=max(high, low + 1e-6))


def labels(ax, image):
    """Display original integer label values, keeping background transparent."""
    ax.imshow(np.ma.masked_where(image == 0, image), cmap="tab20", interpolation="nearest")


def centered_crop(array, y, x, fill=0):
    """Keep the original centroid at (128,128), padding only unavailable source pixels."""
    y0, x0 = y - 128, x - 128
    ya, yb = max(0, y0), min(array.shape[-2], y + 128)
    xa, xb = max(0, x0), min(array.shape[-1], x + 128)
    result = np.full((*array.shape[:-2], 256, 256), fill, dtype=array.dtype)
    result[..., ya - y0 : yb - y0, xa - x0 : xb - x0] = array[..., ya:yb, xa:xb]
    return result


def save_review(folder, name, mips, tracks, cells, nuclei, cytoplasm, members, anchor):
    """Make exact 256-native-pixel, track-centered six-panel review figures."""
    y, x = round(anchor.y), round(anchor.x)
    ys, xs = slice(y - 128, y + 128), slice(x - 128, x + 128)
    valid = centered_crop(np.ones(tracks.shape, dtype=np.uint8), y, x)
    fig, axes = plt.subplots(2, 3, figsize=(9, 6.5))
    for ax, channel in zip(axes[0], ("nuclei_prediction", "membrane_prediction", GFP)):
        gray(ax, centered_crop(mips[channel].astype(np.float32), y, x, np.nan))
        edge(ax, centered_crop(tracks, y, x), "cyan")
        if channel == "nuclei_prediction":
            edge(ax, centered_crop(nuclei, y, x), "magenta")
        else:
            edge(ax, centered_crop(cells, y, x), "magenta")
        ax.set_title(channel, fontsize=9)
    for ax, array, title in zip(
        axes[1],
        (cells, nuclei, cytoplasm),
        ("Membrane cell instances", "Separate CP-DINO nuclear instances", "Cells minus ALL CP-DINO nuclear pixels"),
    ):
        labels(ax, centered_crop(array, y, x))
        edge(ax, centered_crop(tracks, y, x), "cyan")
        ax.set_title(title, fontsize=9)
    for ax in axes.flat:
        ax.imshow(np.ma.masked_where(valid > 0, np.ones(valid.shape)), cmap=ListedColormap(["#aaaaaa"]))
        for row in members.itertuples():
            if xs.start <= row.x < xs.stop and ys.start <= row.y < ys.stop:
                ax.plot(row.x - xs.start, row.y - ys.start, "o", mfc="none", mec="yellow", ms=5)
                ax.text(row.x - xs.start + 4, row.y - ys.start, str(row.track_id), color="yellow", fontsize=7)
        ax.axis("off")
    fig.suptitle(
        f"{name}: 256 native pixels centered on original track {anchor.track_id}\n"
        "Cyan = saved tracking; magenta = inferred; gray = unavailable source pixels. Independent panel stretch.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=2.5)
    fig.savefig(folder / f"{name}.png", dpi=180)
    fig.savefig(folder / f"{name}.pdf")
    plt.close(fig)


def axial_review(folder, position, source, status, anchor, tracks, mips):
    """Inspect wider native z support without inferring floating state from 2D images."""
    y, x = round(anchor.y), round(anchor.x)
    ys, xs = slice(y - 128, y + 128), slice(x - 128, x + 128)
    focus = int(status["z_start"]) + 1
    channels = ("Phase3D", "nuclei_prediction", "membrane_prediction", GFP)
    clipped_y = slice(max(0, ys.start), min(position.shape[-2], ys.stop))
    clipped_x = slice(max(0, xs.start), min(position.shape[-1], xs.stop))
    pad = (
        (0, 0),
        (clipped_y.start - ys.start, ys.stop - clipped_y.stop),
        (clipped_x.start - xs.start, xs.stop - clipped_x.stop),
    )
    slabs = {
        ch: np.pad(
            np.asarray(source[anchor.t, position.channels.index(ch), :, clipped_y, clipped_x]),
            pad,
            constant_values=np.nan,
        )
        for ch in channels
    }
    indices = [focus + k for k in (-12, -8, -4, 0, 4, 8, 12) if 0 <= focus + k < position.shape[2]]
    np.savez_compressed(
        folder / "axial_crop_all_z.npz",
        **slabs,
        focus_index=focus,
        native_y0=ys.start,
        native_x0=xs.start,
        spacing_zyx_um=np.array(position.scale[2:]),
    )
    fig, axes = plt.subplots(4, len(indices), figsize=(2 * len(indices), 8), squeeze=False)
    for panel, ch in zip(axes, channels):
        limits = np.nanpercentile(slabs[ch], [1, 99.5])
        for ax, z in zip(panel, indices):
            gray(ax, slabs[ch][z], limits)
            ax.plot(128, 128, "o", mfc="none", mec="yellow", ms=5)
            ax.set_title(f"{ch.split()[0]}, z{z} ({(z - focus) * position.scale[2]:+.2f} µm)", fontsize=8)
            ax.axis("off")
    fig.suptitle(
        "Fixed 256-native-pixel XY; fixed channel stretch across native planes\n"
        "Selected planes test axial support; these images do not establish whether a cell is floating.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(folder / "axial_planes.png", dpi=180)
    plt.close(fig)
    wide_start, wide_stop = max(0, focus - 8), min(position.shape[2], focus + 9)
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for j, ch in enumerate(channels):
        narrow = centered_crop(mips[ch].astype(np.float32), y, x, np.nan)
        wide = slabs[ch][wide_start:wide_stop].max(axis=0)
        limits = np.nanpercentile(wide, [1, 99.5])
        for ax, image, title in ((axes[0, j], narrow, "Locked 5 planes"), (axes[1, j], wide, "Wider 17 planes")):
            gray(ax, image, limits)
            edge(ax, centered_crop(tracks, y, x) > 0, "cyan")
            ax.plot(128, 128, "o", mfc="none", mec="yellow", ms=5)
            ax.set_title(f"{title}: {ch.split()[0]}", fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(folder / "axial_mip_comparison.png", dpi=180)
    plt.close(fig)
    profiles = []
    for tid in (51, 169):
        footprint = centered_crop(tracks, y, x) == tid
        if footprint.any():
            for ch in channels:
                for z, plane in enumerate(slabs[ch]):
                    profiles.append(
                        {
                            "track_id": tid,
                            "channel": ch,
                            "z": z,
                            "z_offset_um": (z - focus) * position.scale[2],
                            **intensity_stats(plane, footprint),
                        }
                    )
    pd.DataFrame(profiles).to_csv(folder / "axial_profiles.csv", index=False)
    return {
        "focus": focus,
        "native_z_um": position.scale[2],
        "selected_planes": indices,
        "wide_z_start": wide_start,
        "wide_z_stop": wide_stop,
        "profile_support": "saved tracking pixels inside 256px crop",
    }


def diagnose_frame(attempt, pilot, fov, t, target_tracks, anchor_id, segmenter):
    """Run four full-FOV inferences and preserve all labels and tracking identities."""
    config = json.loads((pilot / "manifest.json").read_text())["scientific"]["config"]
    folder = attempt / config["dataset"] / fov / f"t{t:06d}"
    folder.mkdir(parents=True, exist_ok=True)
    original = pilot / "frames" / fov / f"t{t:06d}"
    members = pd.read_parquet(original / "membership.parquet")
    members.to_parquet(folder / "original_membership.parquet", index=False)
    status = json.loads((original / "status.json").read_text())
    dump(folder / "original_status.json", status)
    position = read_position(Path(config["source"]) / fov)
    source = position.open()
    channels = ("Phase3D", "nuclei_prediction", "membrane_prediction", GFP)
    saved_inputs = folder / "original_and_inputs.npz"
    if saved_inputs.exists():
        with np.load(saved_inputs) as archive:
            mips = {ch: archive[ch] for ch in channels}
    else:
        mips = {
            ch: np.asarray(source[t, position.channels.index(ch), slice(status["z_start"], status["z_stop"])]).max(
                axis=0
            )
            for ch in channels
        }
    group = zarr.open_group(str(pilot / "segmentations.ome.zarr" / fov), mode="r")
    tracks = np.asarray(group["labels/nuclei/0"][t, 0, 0], dtype=np.uint32)
    old_cells = np.asarray(group["0"][t, 0, 0], dtype=np.uint32)
    old_cyto = np.asarray(group["labels/cytoplasm/0"][t, 0, 0], dtype=np.uint32)
    np.savez_compressed(
        folder / "original_and_inputs.npz",
        tracked_nuclei=tracks,
        original_cells=old_cells,
        original_cytoplasm=old_cyto,
        **mips,
    )
    anchor = members.loc[members.track_id.eq(anchor_id)].iloc[0]
    assert int(anchor.t) == t
    reports = []
    for name, target in (("native", None), ("058um", 0.58)):
        saved = folder / f"{name}_labels.npz"
        if saved.exists():
            with np.load(saved) as archive:
                cells, nuclei = archive["cells"], archive["inferred_nuclei"]
                working_cells, working_nuclei = archive["working_cells"], archive["working_nuclei"]
            cell_timing = {"reused_saved_inference": True, "working_shape": list(working_cells.shape)}
            nuclear_timing = {"reused_saved_inference": True, "working_shape": list(working_nuclei.shape)}
        else:
            cells, working_cells, cell_timing = stock_inference(
                segmenter, mips["membrane_prediction"], position.pixel_size_yx_um, target
            )
            nuclei, working_nuclei, nuclear_timing = stock_inference(
                segmenter, mips["nuclei_prediction"], position.pixel_size_yx_um, target
            )
        cytoplasm = cells.copy()
        cytoplasm[nuclei > 0] = 0
        tracked_cytoplasm = cells.copy()
        tracked_cytoplasm[tracks > 0] = 0
        assert np.array_equal(cytoplasm[nuclei == 0], cells[nuclei == 0])
        np.savez_compressed(
            folder / f"{name}_labels.npz",
            cells=cells,
            inferred_nuclei=nuclei,
            cytoplasm=cytoplasm,
            cytoplasm_using_tracking=tracked_cytoplasm,
            working_cells=working_cells,
            working_nuclei=working_nuclei,
        )
        for tid in target_tracks:
            row = members.loc[members.track_id.eq(tid)].iloc[0]
            footprint = tracks == row.nucleus_label
            assert footprint.any() and tracks[round(row.y), round(row.x)] == row.nucleus_label
            cell_matches = overlap_instances(footprint, cells, row.y, row.x)
            nuclear_matches = overlap_instances(footprint, nuclei, row.y, row.x)
            cell_id = cell_matches[0]["instance_id"] if cell_matches else 0
            report = {
                "dataset": config["dataset"],
                "fov": fov,
                "t": t,
                "scale": name,
                "target_pixel_size_um": target,
                "track_id": tid,
                "original_nucleus_label": int(row.nucleus_label),
                "original_track_centroid_yx": [float(row.y), float(row.x)],
                "original_cell_id": int(row.cell_id),
                "original_overlap_fraction": float(row.overlap_fraction),
                "original_nuclei_in_cell": int(row.nuclei_in_cell),
                "cell_overlaps": cell_matches,
                "nuclear_overlaps": nuclear_matches,
                "nuclear_background_fraction": float((footprint & (nuclei == 0)).sum() / footprint.sum()),
                "inferred_nucleus_at_track_centroid": int(nuclei[round(row.y), round(row.x)]),
                "inferred_cell_at_track_centroid": int(cells[round(row.y), round(row.x)]),
                "full_fov_cell_count": int(len(np.unique(cells)) - (0 in cells)),
                "full_fov_nuclear_count": int(len(np.unique(nuclei)) - (0 in nuclei)),
                "membrane_inference": cell_timing,
                "nuclear_inference": nuclear_timing,
            }
            if cell_id:
                cell_mask = cells == cell_id
                near_nuclei = overlap_instances(cell_mask, nuclei, row.y, row.x)
                report.update(
                    {
                        "dominant_cell_id": cell_id,
                        "inferred_nuclei_overlapping_dominant_cell": near_nuclei,
                        "inferred_nuclei_majority_inside_dominant_cell": sum(
                            n["fraction_of_inferred_instance"] > 0.5 for n in near_nuclei
                        ),
                        "cytoplasm_with_separate_nuclei": intensity_stats(mips[GFP], cytoplasm == cell_id),
                        "cytoplasm_with_tracked_nuclei": intensity_stats(mips[GFP], tracked_cytoplasm == cell_id),
                        "removed_by_inferred_nuclei_px": int((cell_mask & (nuclei > 0)).sum()),
                        "removed_by_tracked_nuclei_px": int((cell_mask & (tracks > 0)).sum()),
                    }
                )
            reports.append(report)
        save_review(folder, name, mips, tracks, cells, nuclei, cytoplasm, members, anchor)
        print("INFERRED", config["dataset"], fov, t, name, cell_timing, nuclear_timing, flush=True)
    if anchor_id == 51:
        axial = axial_review(folder, position, source, status, anchor, tracks, mips)
        dump(folder / "axial_summary.json", axial)
    dump(folder / "comparison.json", {"tracks": reports})
    return reports


def main():
    """Execute only the three diagnostic frames in a newly claimed July attempt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="Preclaimed new diagnostic directory containing claim.json only"
    )
    parser.add_argument("--include-april", action="store_true", help="Also run the single April t38 comparator")
    parser.add_argument("--resume", action="store_true", help="Reuse saved masks; archive the rendering fix")
    args = parser.parse_args()
    assert_compute_node()
    assert torch.cuda.is_available()
    attempt = args.output
    if not args.resume:
        assert list(attempt.iterdir()) == [attempt / "claim.json"], "Refuse to overwrite any diagnostic artifacts"
    repo = Path(__file__).resolve().parents[3]
    provenance = attempt / "provenance"
    if args.resume:
        provenance = provenance / "rendering_fix_01"
    provenance.mkdir()
    shutil.copy2(__file__, provenance / Path(__file__).name)
    paths = [Path(__file__), *sorted((repo / "applications/dynaclr/src/dynaclr/phenotyping").glob("*.py"))]
    hashes = {}
    for path in paths:
        relative = path.relative_to(repo)
        archived = provenance / "source" / relative
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, archived)
        hashes[str(relative)] = sha256_file(archived)
    for name in ("PLAN.md", "00_pilot.md"):
        source = (
            Path("/home/eduardo.hirata/repos/ed-planning/projects/dynaclr/segmentation/cellpose-dino-dynacell") / name
        )
        shutil.copy2(source, provenance / name)
        hashes[name] = sha256_file(source)
    references = {}
    for name in ("segmentation_cpdino.py", "pipeline_cache.py", "segmentation.py"):
        git_path = f"applications/dynacell/src/dynacell/evaluation/{name}"
        content = subprocess.check_output(["git", "show", f"{REFERENCE}:{git_path}"], cwd=repo)
        (provenance / name).write_bytes(content)
        references[git_path] = hashlib.sha256(content).hexdigest()
    reference_commit = subprocess.check_output(["git", "rev-parse", REFERENCE], cwd=repo, text=True).strip()
    cases = [
        (JULY / "01_annotation_well_scope", "A/3/001001", 64, (51, 169), 51),
        (JULY / "01_annotation_well_scope", "A/3/001001", 65, (51, 169), 51),
    ]
    if args.include_april:
        cases.append((APRIL, "A/2/0001", 38, (62, 91, 94), 94))
    config = {
        "frames": [
            {"pilot": str(p), "fov": f, "t": t, "tracks": list(ids), "anchor": anchor} for p, f, t, ids, anchor in cases
        ],
        "inference_scope": "full native FOV; display crops only are 256 native pixels",
        "projection": "original per-frame Phase3D focus-1:focus+4 (5-plane MIP)",
        "working_pixel_sizes_um": [None, 0.58],
        "backend": "stock CellposeModel.eval, not cubic",
        "reference_difference": (
            "same raw grayscale normalization and inference thresholds; stock tiled implementation, "
            "MIP instead of reference in-focus slice; .58um arm adds downsampling"
        ),
        "carving": (
            "independent nucleus and membrane inference; remove all nuclear pixels; no cell splitting/relabeling"
        ),
        "reference_commit": reference_commit,
    }
    dump(attempt / "config.json", config)
    for pilot in {c[0] for c in cases}:
        dest = provenance / pilot.parent.parent.parent.name
        dest.mkdir()
        shutil.copy2(pilot / "manifest.json", dest / "original_manifest.json")
        shutil.copy2(pilot / "pilot.yml", dest / "pilot.yml")
    segmenter = CPDinoSegmenter(Segmentation())
    environment = {
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "packages": {
            name: importlib.metadata.version(name) for name in ("cellpose", "torch", "numpy", "scikit-image", "zarr")
        },
        "model": segmenter.identity,
        "eval_signature": str(inspect.signature(segmenter.model.eval)),
        "git_base": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "source_sha256": hashes,
        "reference_files_sha256": references,
    }
    dump(provenance / "environment.json", environment)
    reports = []
    for case in cases:
        reports.extend(diagnose_frame(attempt, *case, segmenter))
        dump(attempt / "progress.json", {"completed_track_scale_rows": len(reports)})
    dump(
        attempt / "summary.json",
        {
            "frames": len(cases),
            "inference_calls": 4 * len(cases),
            "tracks": reports,
            "production_changes": False,
            "automatic_splitting": False,
            "interpretation": (
                "Separate nuclear inference can change carving but does not split membrane instances. "
                "All overlapping nuclear instances retained; 2D images do not establish floating state."
            ),
        },
    )
    print("COMPLETE", attempt, flush=True)


if __name__ == "__main__":
    main()
