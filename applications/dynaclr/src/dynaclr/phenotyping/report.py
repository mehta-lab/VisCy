"""Review artifacts independent of embeddings; unreviewed cells are not positives."""

from pathlib import Path

import numpy as np
import pandas as pd


def frame_overlay(path, membrane, virtual_nucleus, fluorescence, cells, nuclei, membership, annotations):
    """Render explicit magenta cell and cyan nuclear boundaries with annotated tracks."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from skimage.segmentation import find_boundaries

    images = [("Virtual membrane", membrane), ("Virtual nucleus", virtual_nucleus)] + list(fluorescence.items())
    fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 7), squeeze=False)
    cell_edges = np.ma.masked_where(~find_boundaries(cells), np.ones(cells.shape))
    nuclear_edges = np.ma.masked_where(~find_boundaries(nuclei), np.ones(nuclei.shape))
    empty = set(annotations.loc[annotations.annotated_empty.fillna(False), "track_id"]) if len(annotations) else set()
    for axis, (name, image) in zip(axes[0], images):
        low, high = np.percentile(image, [1, 99.5])
        axis.imshow(image, cmap="gray", vmin=low, vmax=max(high, low + 1e-6))
        axis.imshow(cell_edges, cmap=ListedColormap(["magenta"]), vmin=0, vmax=1, alpha=0.7)
        axis.imshow(nuclear_edges, cmap=ListedColormap(["cyan"]), vmin=0, vmax=1, alpha=0.7)
        for row in membership.itertuples():
            if row.track_id in empty:
                axis.plot(row.x, row.y, "o", markersize=10, markerfacecolor="none", markeredgecolor="yellow")
                axis.text(row.x, row.y, str(row.track_id), color="yellow", fontsize=7)
        axis.set_title(name, fontsize=9)
        axis.axis("off")
    fig.suptitle("Cell boundaries: magenta; nuclei: cyan; annotated empty tracks: yellow", fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def diagnostic_tables(observations, comparison):
    """Compare annotated intensity ranks and exploratory candidate cutoff sweeps."""
    ranks, curves = [], []
    if observations.empty:
        return pd.DataFrame(), pd.DataFrame()
    for (dataset, marker), population in observations[observations.valid_signal.fillna(False)].groupby(
        ["dataset", "marker"]
    ):
        for metric in ("mean_intensity", "median_intensity", "p90_intensity"):
            if metric not in population:
                continue
            reference = np.sort(population[metric].dropna().to_numpy())
            if not len(reference) or "marker" not in comparison:
                continue
            selected = comparison[
                (comparison.dataset == dataset)
                & (comparison.marker == marker)
                & comparison.annotated_empty.fillna(False)
                & (comparison.annotation_match_status == "matched")
            ]
            for confidence, marked in selected.groupby("annotation_confidence"):
                values = marked[metric].dropna().to_numpy()
                for q in (0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75):
                    cutoff = float(np.quantile(reference, q))
                    curves.append(
                        dict(
                            dataset=dataset,
                            marker=marker,
                            metric=metric,
                            confidence=confidence,
                            reference_quantile=q,
                            candidate_threshold=cutoff,
                            measured_empty_frames=len(values),
                            fraction_flagged=float(np.mean(values < cutoff)) if len(values) else np.nan,
                            threshold_status="exploratory_not_selected_or_independently_validated",
                        )
                    )
                for row in marked.itertuples():
                    value = getattr(row, metric)
                    if np.isfinite(value):
                        rank = (
                            50
                            * (
                                np.searchsorted(reference, value, side="left")
                                + np.searchsorted(reference, value, side="right")
                            )
                            / len(reference)
                        )
                        ranks.append(
                            dict(
                                dataset=dataset,
                                fov=row.fov,
                                track_id=row.track_id,
                                t=row.t,
                                marker=marker,
                                confidence=confidence,
                                metric=metric,
                                intensity=value,
                                population_percentile=rank,
                            )
                        )
    return pd.DataFrame(ranks), pd.DataFrame(curves)


def summary_report(output: Path, cells, observations, comparison, status, membership):
    """Write cell-size plots and independent annotation signal diagnostics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from dynaclr.phenotyping.io import atomic_json, atomic_parquet

    qc = output / "qc"
    qc.mkdir(exist_ok=True)
    report = {
        "processed_frames": int(len(status)),
        "frame_states": status.state.value_counts().to_dict() if len(status) else {},
        "assignment_states": membership.assignment_status.value_counts().to_dict() if len(membership) else {},
        "threshold_status": "none_active",
        "unmarked_annotations": "unreviewed",
        "annotation_counts": comparison.annotation_match_status.value_counts().to_dict() if len(comparison) else {},
        "cell_size_percentiles": {},
    }
    for col in ("area_px", "area_um2", "bbox_height_px", "bbox_width_px"):
        if col in cells and len(cells):
            report["cell_size_percentiles"][col] = {
                str(q): float(cells[col].quantile(q)) for q in (0.5, 0.9, 0.95, 0.99)
            }
    if "nuclear_centered_patch_side_px" in observations:
        patches = observations.drop_duplicates(["dataset", "fov", "t", "track_id"])[
            "nuclear_centered_patch_side_px"
        ].dropna()
        report["nuclear_centered_patch_side_percentiles"] = (
            {str(q): float(patches.quantile(q)) for q in (0.5, 0.9, 0.95, 0.99)} if len(patches) else {}
        )
    ranks, curves = diagnostic_tables(observations, comparison)
    atomic_parquet(qc / "annotation_signal_ranks.parquet", ranks)
    atomic_parquet(qc / "exploratory_threshold_sweep.parquet", curves)
    if len(cells):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(cells.area_um2, bins=50)
        axes[0].set(xlabel="Cell area (µm²)", ylabel="Cells / frames")
        axes[1].hist(np.maximum(cells.bbox_height_px, cells.bbox_width_px), bins=50)
        axes[1].set(xlabel="Bounding square side (native pixels)", ylabel="Cells / frames")
        fig.tight_layout()
        fig.savefig(qc / "cell_sizes.png", dpi=150)
        plt.close(fig)
    if len(observations):
        for marker, pop in observations[observations.valid_signal.fillna(False)].groupby("marker"):
            fig, axes = plt.subplots(1, 3, figsize=(13, 4))
            for axis, metric in zip(axes, ("mean_intensity", "median_intensity", "p90_intensity")):
                values = pop[metric].dropna()
                if not len(values):
                    continue
                bins = np.linspace(values.min(), max(values.max(), values.min() + 1e-6), 60)
                axis.hist(values, bins=bins, density=True, alpha=0.4, label="Measured population (unreviewed)")
                marked = (
                    comparison[
                        (comparison.marker == marker)
                        & comparison.annotated_empty.fillna(False)
                        & (comparison.annotation_match_status == "matched")
                    ]
                    if "marker" in comparison
                    else pd.DataFrame()
                )
                for confidence, group in marked.groupby("annotation_confidence") if len(marked) else []:
                    axis.hist(
                        group[metric].dropna(),
                        bins=bins,
                        density=True,
                        histtype="step",
                        label=f"Annotated empty ({confidence})",
                    )
                axis.set(xlabel=f"Cytoplasm {metric} (raw units)", ylabel="Density")
            axes[0].legend(fontsize=7)
            fig.suptitle(marker)
            fig.tight_layout()
            safe = "".join(c if c.isalnum() else "_" for c in marker)
            fig.savefig(qc / f"signal_{safe}.png", dpi=150)
            plt.close(fig)
    atomic_json(qc / "summary.json", report)
    (qc / "README.md").write_text(
        "# Cell phenotyping review\n\nRaw cytoplasmic measurements are independent of embeddings. "
        "Unmarked annotations are unreviewed; specificity and precision cannot be estimated. "
        "Annotation ranks and empirical threshold sweeps are exploratory. "
        "Eligibility remains unknown pending threshold review. "
        "Frame overlays identify annotated tracks in yellow.\n"
    )
    return report
