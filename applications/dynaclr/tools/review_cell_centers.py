"""Compare three centers at 256 original pixels using existing April masks only."""

import argparse
import inspect
import json
import platform
import re
import shutil
import sys
from contextlib import closing
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import zarr
from review_cytoplasm_coverage import bounds, count_inside, file_sha, padded_crop
from scipy.ndimage import binary_dilation
from skimage.segmentation import find_boundaries

from dynaclr.phenotyping import measurements
from dynaclr.phenotyping.io import assert_compute_node, read_position

matplotlib.use("Agg")
MODES = {
    "nuclear": ("Nuclear", "#ffab40"),
    "bbox": ("Cell bbox", "#00b8f0"),
    "cytoplasm": ("Uniform cytoplasm COM", "#f050a0"),
}
KEYS = ["dataset", "fov", "t", "track_id"]
CELL_KEYS = ["dataset", "fov", "t", "cell_id"]


def render(source, config, row, masks, target):
    """Show full Phase context/instance contours, then Phase and fluorescence zooms of three centers."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    fov, t = row.fov, int(row.t)
    status = json.loads((source / "frames" / fov / f"t{t:06d}" / "status.json").read_text())
    position = read_position(Path(config["source"]) / fov)
    assert np.allclose(position.pixel_size_yx_um, (0.1133, 0.1133), rtol=0, atol=1e-9)
    y0, x0, y1, x1 = bounds(row.nuclear_center_y, row.nuclear_center_x, 512)
    context = (y0, x0, y1, x1)
    ys = slice(max(0, y0), min(position.shape[-2], y1))
    xs = slice(max(0, x0), min(position.shape[-1], x1))
    slab = slice(status["z_start"], status["z_stop"])
    assert slab.stop - slab.start == 5
    print(f"READ full Phase {fov} t{t}: five native planes", flush=True)
    with closing(zarr.storage.LocalStore(str(position.path), read_only=True)) as store:
        array = zarr.open_group(store=store, mode="r")[position.array_path]
        phase = np.asarray(array[t, position.channels.index(config["focus_channel"]), slab]).max(axis=0)
        images = []
        channels = [
            config["membrane_channel"],
            next(c["name"] for c in config["fluorescence"] if c["marker"] == "TOMM20"),
        ]
        for channel in channels:
            pixels = np.asarray(array[t, position.channels.index(channel), slab, ys, xs]).max(axis=0)
            low, high = np.percentile(pixels, (1, 99))
            display = np.zeros((512, 512), dtype=float)
            display[ys.start - y0 : ys.stop - y0, xs.start - x0 : xs.stop - x0] = np.clip(
                (pixels - low) / max(float(high - low), 1e-9), 0, 1
            )
            images.append(display)
    low, high = np.percentile(phase, (1, 99))
    phase_display = np.clip((phase - low) / max(float(high - low), 1e-9), 0, 1)
    overlay = np.repeat((0.65 * phase_display)[..., None], 3, axis=-1)
    # Integer labels retain boundaries between touching instances, unlike a binary union.
    overlay[binary_dilation(find_boundaries(masks["cells"], mode="inner"), iterations=2)] = [0.3, 1, 0.45]
    overlay[binary_dilation(find_boundaries(masks["nuclei"], mode="inner"), iterations=2)] = [1, 1, 1]
    fig = plt.figure(figsize=(18, 17))
    grid = fig.add_gridspec(3, 6, height_ratios=(1.3, 1, 1), hspace=0.22, wspace=0.16)
    top = [fig.add_subplot(grid[0, :3]), fig.add_subplot(grid[0, 3:])]
    titles = (
        "Full FOV: raw Phase3D five-plane MIP",
        "Same Phase, dimmed: INTEGER cell (green) / nuclear (white) contours",
    )
    for axis, pixels, title in zip(top, (phase_display, overlay), titles, strict=True):
        axis.imshow(pixels, cmap="gray", origin="upper")
        axis.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), 512, 512, fill=False, edgecolor="yellow", linewidth=1.4))
        axis.set_xlim(-0.5, phase.shape[1] - 0.5)
        axis.set_ylim(phase.shape[0] - 0.5, -0.5)
        axis.set_title(title, fontsize=12)
        axis.axis("off")
    cell, nuclei = padded_crop(masks["cells"], context), padded_crop(masks["nuclei"], context)
    target_edges = find_boundaries(cell == row.cell_id, mode="inner")
    nuclear_edges = find_boundaries(nuclei, mode="inner")
    phase_zoom = np.repeat((0.65 * padded_crop(phase_display, context))[..., None], 3, axis=-1)
    phase_zoom[find_boundaries(cell, mode="inner")] = [0.1, 0.55, 0.2]
    phase_zoom[binary_dilation(target_edges, iterations=1)] = [0.3, 1, 0.45]
    phase_zoom[nuclear_edges] = [1, 1, 1]
    composite = np.repeat((0.5 * images[0])[..., None], 3, axis=-1)
    composite[..., 1] = np.clip(composite[..., 1] + 0.5 * images[1], 0, 1)
    composite[target_edges] = [0.3, 1, 0.45]
    composite[nuclear_edges] = [1, 1, 1]
    for grid_row, zoom in ((1, phase_zoom), (2, composite)):
        for index, (mode, (label, color)) in enumerate(MODES.items()):
            axis = fig.add_subplot(grid[grid_row, 2 * index : 2 * index + 2])
            axis.imshow(zoom, origin="upper")
            for other, (_, marker_color) in MODES.items():
                axis.plot(
                    int(row[f"{other}_center_x"]) - x0,
                    int(row[f"{other}_center_y"]) - y0,
                    "+",
                    color=marker_color,
                    markersize=10,
                    markeredgewidth=2,
                )
            by0, bx0, _, _ = bounds(row[f"{mode}_center_y"], row[f"{mode}_center_x"], 256)
            axis.add_patch(
                Rectangle((bx0 - x0 - 0.5, by0 - y0 - 0.5), 256, 256, fill=False, edgecolor=color, linewidth=2)
            )
            axis.set_xlim(-0.5, 511.5)
            axis.set_ylim(511.5, -0.5)
            axis.set_title(
                f"{label}: exact 256-pixel square\nCytoplasm retained: {row[f'fraction_256_{mode}']:.2%}", fontsize=12
            )
            axis.axis("off")
    legend = [
        Line2D([0], [0], color=color, marker="+", linestyle="none", markersize=10, label=label)
        for label, color in MODES.values()
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, fontsize=12)
    fig.suptitle(
        f"{row.example_role}\nApril {fov} track {int(row.track_id)}, t{t}; cell {int(row.cell_id)}, "
        f"assigned nuclei={int(row.nuclei_in_cell)}; "
        f"bbox {int(row.bbox_x1 - row.bbox_x0)}×{int(row.bbox_y1 - row.bbox_y0)} px\n"
        "Top: full-FOV Phase3D context (yellow: 512-native box). Middle: same context, Phase zoom with all cell "
        "(green; target bright) and nuclear (white) instance contours.\nBottom: same context, membrane gray + GFP "
        "green. One box per center. Display stretch only; unclamped boxes; no model or survival claim.",
        fontsize=13,
    )
    fig.subplots_adjust(top=0.90, bottom=0.05)
    stem = f"{fov.replace('/', '_')}_track{int(row.track_id)}_t{t:03d}"
    fig.savefig(target / f"{stem}.png", dpi=170)
    fig.savefig(target / f"{stem}.pdf")
    plt.close(fig)
    return {
        "file": stem,
        "fov": fov,
        "t": t,
        "track_id": int(row.track_id),
        "context_bounds": list(context),
        "z_start": slab.start,
        "z_stop": slab.stop,
    }


def review(source, slug, target=None):
    """Write one fresh numbered attempt with paired coverage and the same five April pages."""
    assert_compute_node()
    assert re.fullmatch(r"[a-z0-9]+(_[a-z0-9]+)*", slug), f"Slug must be snake_case: {slug}"
    prior = source.parent / "04_cytoplasm_coverage"
    data = pd.read_csv(prior / "anchor_coverage.csv")
    examples = pd.read_csv(prior / "example_observations.csv")
    boolean_columns = ("population_frame", "coverage_valid", "touches_border")
    for column in boolean_columns:
        assert data[column].dropna().isin([True, False]).all()
        data[column] = data[column].fillna(column == "touches_border").astype(bool)
    assert not data.duplicated(KEYS).any() and len(examples) == 5
    population = data[data.population_frame]
    assert len(population[["fov", "t"]].drop_duplicates()) == 18 and set(population.t) == {2, 30, 60}
    manifest = json.loads((source / "manifest.json").read_text())
    config = manifest["scientific"]["config"]
    assert config["dataset"] == "2026_04_10_A549_TOMM20_ZIKV"
    center_fn = measurements.cell_centers
    if target is not None:
        assert target.parent == source.parent and target.is_dir() and not (target / "summary.json").exists()
        assert re.fullmatch(rf"\d+_{slug}", target.name)
    while target is None:
        numbers = [
            int(m.group(1)) for p in source.parent.iterdir() if p.is_dir() and (m := re.match(r"^(\d+)_", p.name))
        ]
        candidate = source.parent / f"{max(numbers, default=-1) + 1:02d}_{slug}"
        try:
            candidate.mkdir(exist_ok=False)
            target = candidate
        except FileExistsError:
            continue
    print(f"RESERVED {target}; helper: {center_fn.__module__}.{center_fn.__name__}", flush=True)
    (target / "provenance").mkdir(exist_ok=True)
    shutil.copy2(__file__, target / "provenance/review_cell_centers.py")
    shutil.copy2(
        Path(__file__).with_name("review_cytoplasm_coverage.py"), target / "provenance/review_cytoplasm_coverage.py"
    )
    (target / "provenance/center_function.py").write_text(inspect.getsource(center_fn))
    inputs = (source / "manifest.json", prior / "anchor_coverage.csv", prior / "example_observations.csv")
    hashes = {str(p): file_sha(p) for p in inputs}
    provenance = {
        "source_attempt": str(source),
        "prior_review": str(prior),
        "slug": slug,
        "argv": sys.argv,
        "hostname": platform.node(),
        "python": sys.executable,
        "script_sha256": file_sha(Path(__file__)),
        "input_sha256": hashes,
        "source_frozen_code": manifest["code"],
        "source_config": config,
        "center_helper": f"{center_fn.__module__}.{center_fn.__name__}",
        "native_pixel_um": 0.1133,
        "square_rule": "int(center), half=128, half-open bounds; no border clamp",
        "cytoplasm_COM": "uniform pixel weights, not intensity weighted; can lie outside a concave mask",
    }
    (target / "provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    data["nuclear_center_y"], data["nuclear_center_x"] = data.y, data.x
    with closing(zarr.storage.LocalStore(str(source / "segmentations.ome.zarr"), read_only=True)) as store:
        root = zarr.open_group(store=store, mode="r")
        for (fov, t), rows in data.groupby(["fov", "t"], sort=True):
            cells, cyto = [np.asarray(root[fov][p][t, 0, 0]) for p in ("0", "labels/cytoplasm/0")]
            centers = center_fn(cells, cyto).set_index("cell_id")
            assert centers.index.is_unique
            for index, row in rows[rows.coverage_valid].iterrows():
                cell_id = int(row.cell_id)
                total = int(np.count_nonzero(cyto == cell_id))
                assert total == int(row.cytoplasm_area_px_exact)
                center = centers.loc[cell_id]
                for mode, prefix in (("bbox", "bbox_center"), ("cytoplasm", "cytoplasm_centroid")):
                    data.loc[index, f"{mode}_center_y"] = center[f"{prefix}_y"]
                    data.loc[index, f"{mode}_center_x"] = center[f"{prefix}_x"]
                for mode in MODES:
                    y, x = data.loc[index, [f"{mode}_center_y", f"{mode}_center_x"]]
                    numerator = count_inside(cyto, cell_id, bounds(y, x, 256))
                    data.loc[index, f"pixels_256_{mode}"] = numerator
                    data.loc[index, f"fraction_256_{mode}"] = numerator / total
                assert abs(data.loc[index, "fraction_256_nuclear"] - row.fraction_nuclear_256) < 1e-12
            print(f"MEASURED {fov} t{t}: {int(rows.coverage_valid.sum())} matched anchors", flush=True)
        keep = KEYS + [
            "cell_id",
            "nucleus_label",
            "nuclei_in_cell",
            "assignment_status",
            "population_frame",
            "coverage_valid",
            "touches_border",
            "cytoplasm_area_px_exact",
            "bbox_y0",
            "bbox_x0",
            "bbox_y1",
            "bbox_x1",
        ]
        keep += [f"{mode}_center_{axis}" for mode in MODES for axis in ("y", "x")]
        keep += [f"{kind}_256_{mode}" for mode in MODES for kind in ("pixels", "fraction")]
        output = data[keep]
        output.to_parquet(target / "center_coverage.parquet", index=False)
        output.to_csv(target / "center_coverage.csv", index=False)
        loaded = pd.read_parquet(target / "center_coverage.parquet")
        assert all(loaded[c].dtype == bool for c in boolean_columns)
        primary = output[output.population_frame & output.coverage_valid & ~output.touches_border]
        assert len(primary) == 548 and len(primary.drop_duplicates(CELL_KEYS)) == 508
        statistics = []
        for stratum, rows in (
            ("all", primary),
            ("single", primary[primary.nuclei_in_cell == 1]),
            ("multi", primary[primary.nuclei_in_cell > 1]),
        ):
            for mode in MODES:
                values = rows[f"fraction_256_{mode}"]
                statistics.append(
                    {
                        "population": "nonborder_matched",
                        "nuclei": stratum,
                        "center": mode,
                        "anchors": len(rows),
                        "unique_cell_frames": len(rows.drop_duplicates(CELL_KEYS)),
                        "median_fraction": float(values.median()),
                        "n_at_least_80_percent": int((values >= 0.8).sum()),
                        "proportion_at_least_80_percent": float((values >= 0.8).mean()),
                    }
                )
        pd.DataFrame(statistics).to_csv(target / "center_summary.csv", index=False)
        selected = examples[KEYS + ["example_role"]].merge(output, on=KEYS, validate="one_to_one")
        assert len(selected) == 5
        selected.to_parquet(target / "example_centers.parquet", index=False)
        report = {
            "complete": False,
            "statistics": statistics,
            "population_frames": 18,
            "population_anchors": 601,
            "primary_anchors": 548,
            "primary_cell_frames": 508,
            "primary_shared_masks": 35,
            "primary_shared_anchors": 75,
            "targeted_examples_outside_population": 2,
            "square_side_native_px": 256,
            "square_side_um": 29.0048,
            "active_filter": None,
            "parquet_boolean_roundtrip_passed": True,
            "note": "Paired geometry of existing masks; repeated/shared anchors are not independent cells. "
            "Uniform COM is not intensity weighted. No model, boundary-accuracy, or survival conclusion.",
        }
        (target / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print("TABLES SAVED; rendering five pages", flush=True)
        pages = []
        for _, row in selected.iterrows():
            print(f"RENDER {row.fov} t{int(row.t)} track{int(row.track_id)}: full Phase + center zooms", flush=True)
            masks = {
                n: np.asarray(root[row.fov][p][int(row.t), 0, 0])
                for n, p in (("cells", "0"), ("nuclei", "labels/nuclei/0"))
            }
            pages.append(render(source, config, row, masks, target))
            print(f"SAVED {pages[-1]['file']}.png/.pdf", flush=True)
    assert all(file_sha(Path(p)) == digest for p, digest in hashes.items()), "Source inputs changed"
    report.update(complete=True, examples=pages, source_checks_passed=True)
    (target / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(target), **report}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-attempt", type=Path, required=True)
    parser.add_argument("--slug", required=True, help="snake_case name for the new NN_<slug> attempt folder")
    args = parser.parse_args()
    review(args.source_attempt, args.slug)
