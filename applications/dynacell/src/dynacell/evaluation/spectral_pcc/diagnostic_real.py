"""Diagnostic spectra plot for real A549 nuclei data.

Loads one position from the A549 zarr store, extracts mid-Z slices,
generates diagnostic spectra plots, and computes DCR A0 per timepoint.

Usage::

    uv run python -m dynacell.evaluation.spectral_pcc.diagnostic_real
    uv run python -m dynacell.evaluation.spectral_pcc.diagnostic_real position=B/2/0000001
"""

import logging
from pathlib import Path

import hydra
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from iohub.ngff import open_ome_zarr
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def plot_pcc_comparison_real(
    df: pd.DataFrame,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """PCC variants + DCR A0 on twin axis for real data."""
    t = df["timepoint"].values

    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = [
        ("PCC_2D", "PCC", "C3"),
        ("BL_PCC_DCR_2D", "BL_PCC_DCR", "C0"),
        ("Spectral_PCC_2D", "Spectral_PCC", "C1"),
        ("Spectral_PCC_FRCW_2D", "FRCW", "C5"),
        ("Spectral_PCC_FRCW_Frozen_2D", "FRCW_Frozen", "C2"),
    ]
    for col, label, color in metrics:
        if col in df.columns:
            ax.plot(t, df[col], color=color, linewidth=1.5, label=label)

    ax.set_xlabel("Timepoint")
    ax.set_ylabel("PCC")
    ax.grid(True, alpha=0.3)

    # DCR A0 on twin axis
    if "DCR_A0" in df.columns:
        ax2 = ax.twinx()
        ax2.plot(t, df["DCR_A0"], color="C7", linewidth=1.5, linestyle="--", label="DCR A0")
        ax2.set_ylabel("DCR A0", color="C7")
        ax2.tick_params(axis="y", labelcolor="C7")
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    else:
        ax.legend(loc="upper right")

    pos_name = df.attrs.get("position", "")
    ax.set_title(f"A549 Nuclei — {pos_name}" if pos_name else "A549 Nuclei")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    log.info("Saved %s", output_path)


def plot_dcr_a0(
    df: pd.DataFrame,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """DCR A0 and DCR resolution vs timepoint."""
    t = df["timepoint"].values

    fig, ax = plt.subplots(figsize=(8, 4))
    if "DCR_A0" in df.columns:
        ax.plot(t, df["DCR_A0"], "C0-", linewidth=1.5, label="DCR A0")
    ax.set_xlabel("Timepoint")
    ax.set_ylabel("DCR A0")
    ax.grid(True, alpha=0.3)

    # DCR resolution on twin axis (cy/um -> higher = better resolution)
    if "DCR_2D" in df.columns:
        ax2 = ax.twinx()
        ax2.plot(t, df["DCR_2D"], "C3--", linewidth=1, alpha=0.7, label="DCR_2D (cy/um)")
        ax2.set_ylabel("DCR resolution (cy/um)", color="C3")
        ax2.tick_params(axis="y", labelcolor="C3")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    else:
        ax.legend(loc="upper right")

    # Annotate drop/CV for DCR_A0
    if "DCR_A0" in df.columns:
        vals = df["DCR_A0"].values
        mask = np.isfinite(vals)
        if mask.sum() > 1:
            slope, intercept = np.polyfit(t[mask], vals[mask], 1)
            y0 = intercept + slope * t[0]
            yT = intercept + slope * t[-1]
            drop = (y0 - yT) / y0 * 100 if y0 > 0 else 0
            cv = np.std(vals[mask]) / np.mean(vals[mask]) * 100
            ax.set_title(f"DCR A0: drop={drop:.1f}% CV={cv:.1f}%")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    log.info("Saved %s", output_path)


def plot_taper_comparison(
    df: pd.DataFrame,
    nbins_sweep: dict[int, np.ndarray],
    taper_sweep: dict[int, np.ndarray],
    output_path: Path,
    dpi: int = 150,
    title: str | None = None,
) -> None:
    """Taper_low sweep with hard-cutoff baselines."""
    t = df["timepoint"].values

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Baselines (solid) — all plain Spectral_PCC
    ax.plot(t, df["PCC_2D"], color="0.55", ls="-", lw=2.0, label="PCC")
    if 0 in nbins_sweep:
        ax.plot(
            t,
            nbins_sweep[0],
            color="0.25",
            ls="-",
            lw=2.0,
            label="Spectral_PCC (nbins_low=0)",
        )
    if 1 in nbins_sweep:
        ax.plot(
            t,
            nbins_sweep[1],
            color="C7",
            ls="-",
            lw=1.5,
            label="Spectral_PCC (nbins_low=1)",
        )
    if 2 in nbins_sweep:
        ax.plot(
            t,
            nbins_sweep[2],
            color="C0",
            ls="-",
            lw=2.0,
            label="Spectral_PCC (nbins_low=2)",
        )

    # Taper sweep (dashed, colormap)
    cmap = plt.cm.plasma_r
    taper_vals = sorted(taper_sweep.keys())
    n_vals = len(taper_vals)
    for i, tl in enumerate(taper_vals):
        color = cmap(0.15 + 0.75 * i / max(n_vals - 1, 1))
        ax.plot(t, taper_sweep[tl], color=color, ls="--", lw=0.9, label=f"taper_low={tl}")

    ax.set_xlabel("Timepoint")
    ax.set_ylabel("PCC")
    ax.set_title(title or "Spectral PCC — taper_low sweep")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7.5, loc="lower left", ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    log.info("Saved %s", output_path)


_DIAG_CONFIG_DIR = str(Path(__file__).resolve().parents[4] / "configs" / "evaluate" / "spectral_pcc")


@hydra.main(version_base="1.2", config_path=_DIAG_CONFIG_DIR, config_name="diagnostic_real")
def main(cfg: DictConfig) -> None:
    """Generate diagnostic spectra and DCR A0 plots for real A549 data."""
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading position %s from %s...", cfg.position, cfg.input_zarr)
    store = open_ome_zarr(cfg.input_zarr, mode="r")
    pos = store[cfg.position]

    mid_z = pos.data.shape[2] // 2
    n_tp = pos.data.shape[0]
    spacing_2d = list(pos.scale[-2:])
    log.info(
        "  Shape: %s, mid_z=%d, spacing=%s, %d timepoints",
        pos.data.shape,
        mid_z,
        spacing_2d,
        n_tp,
    )

    # Load all mid-Z GT and prediction slices
    log.info("Loading %d mid-Z GT + prediction slices...", n_tp)
    gt_series = np.array(pos.data[:, cfg.gt_channel, mid_z]).astype(np.float32)
    pred_series = np.array(pos.data[:, cfg.pred_channel, mid_z]).astype(np.float32)
    pred_slice = pred_series[0]
    log.info("  GT series shape: %s", gt_series.shape)

    # t=0 as reference (highest SNR)
    clean = gt_series[0]

    # Approximate SNR from mean intensity (for panel titles)
    means = gt_series.mean(axis=(1, 2))
    approx_snr = np.sqrt(np.maximum(means, 0))

    # 1. Diagnostic spectra plot (reuse from simulate_beads)
    from dynacell.evaluation.spectral_pcc.simulate_beads import plot_diagnostic_spectra

    spectral_pcc_kwargs = OmegaConf.to_container(cfg.spectral_pcc, resolve=True)
    log.info("Generating diagnostic spectra plot...")
    plot_diagnostic_spectra(
        clean,
        gt_series,
        pred_slice,
        spacing_2d,
        approx_snr,
        output_dir / "diagnostic_spectra_real.png",
        spectral_pcc_kwargs=spectral_pcc_kwargs,
        n_snapshots=cfg.n_snapshots,
        wavelength_emission=cfg.optics.wavelength_emission,
        numerical_aperture=cfg.optics.numerical_aperture,
    )

    # 2. Compute DCR A0 per timepoint
    from dynacell.evaluation.spectral_pcc.evaluate import compute_gt_reliability

    dcr_kwargs = OmegaConf.to_container(cfg.dcr, resolve=True)
    log.info("Computing DCR A0 for %d timepoints...", n_tp)
    a0_vals = np.zeros(n_tp)
    for t in range(n_tp):
        if (t + 1) % 25 == 0 or t == 0:
            log.info("  timepoint %d / %d", t + 1, n_tp)
        a0, _ = compute_gt_reliability(gt_series[t], spacing_2d, dcr_kwargs)
        a0_vals[t] = a0

    # 3. Load pre-computed metrics + add DCR_A0
    metrics_csv = Path(cfg.metrics_dir) / cfg.position / "metrics.csv"
    if metrics_csv.exists():
        df = pd.read_csv(metrics_csv)
        df["DCR_A0"] = a0_vals[: len(df)]
        df.attrs["position"] = cfg.position

        # 4. PCC comparison with DCR_A0
        plot_pcc_comparison_real(df, output_dir / "pcc_comparison_real.png")

        # 5. DCR_A0 stability plot
        plot_dcr_a0(df, output_dir / "dcr_a0_real.png")

        # 6. Save updated metrics with DCR_A0
        df.to_csv(output_dir / "metrics_with_a0.csv", index=False)
        log.info("Saved %s", output_dir / "metrics_with_a0.csv")

        # --- Precompute mean-filled arrays (same preprocessing as evaluate.py) ---
        from cubic.metrics.bandlimited import spectral_pcc as _spcc

        from dynacell.evaluation.spectral_pcc.evaluate import _prepare_masked_inputs
        from dynacell.evaluation.spectral_pcc.simulate_beads import plot_pcc_comparison

        log.info("Precomputing mean-filled arrays for %d timepoints...", n_tp)
        gt_filled_list = []
        pred_filled_list = []
        for ti in range(n_tp):
            gf, pf, _, _, _ = _prepare_masked_inputs(
                gt_series[ti],
                pred_series[ti],
            )
            gt_filled_list.append(gf)
            pred_filled_list.append(pf)

        # 7. nbins_low sweep
        nbins_low_range = list(range(11))  # 0..10
        log.info(
            "Computing nbins_low sweep (%d values x %d timepoints)...",
            len(nbins_low_range),
            n_tp,
        )

        sweep_values: dict[int, np.ndarray] = {}
        for nbl in nbins_low_range:
            vals = np.empty(n_tp)
            for ti in range(n_tp):
                vals[ti] = _spcc(
                    pred_filled_list[ti],
                    gt_filled_list[ti],
                    spacing=spacing_2d,
                    nbins_low=nbl,
                )
            sweep_values[nbl] = vals
            log.info("  nbins_low=%d done", nbl)

        # Save sweep CSV
        sweep_rows = []
        for nbl, vals in sweep_values.items():
            for ti, v in enumerate(vals):
                sweep_rows.append(
                    {
                        "timepoint": ti,
                        "nbins_low": nbl,
                        "Spectral_PCC_2D": v,
                    }
                )
        pd.DataFrame(sweep_rows).to_csv(
            output_dir / "nbins_low_sweep.csv",
            index=False,
        )
        log.info("Saved %s", output_dir / "nbins_low_sweep.csv")

        # Plot
        plot_pcc_comparison(
            df,
            output_dir / "nbins_low_sweep_real.png",
            sweep_values=sweep_values,
            nbins_low_sweep=nbins_low_range,
            title=f"A549 Nuclei ({cfg.position}) — nbins_low sweep",
            pcc_label="PCC",
        )

        # 8. Soft low-k cosine taper sweep
        taper_range = [0, 1, 2, 3, 4, 5]
        log.info(
            "Computing taper_low sweep (%d values x %d timepoints)...",
            len(taper_range),
            n_tp,
        )

        taper_sweep: dict[int, np.ndarray] = {}
        for tl in taper_range:
            vals = np.empty(n_tp)
            for ti in range(n_tp):
                vals[ti] = _spcc(
                    pred_filled_list[ti],
                    gt_filled_list[ti],
                    spacing=spacing_2d,
                    taper_low=tl,
                )
            taper_sweep[tl] = vals
            log.info("  taper_low=%d done", tl)

        # Save taper sweep CSV
        taper_rows = []
        for tl, vals in taper_sweep.items():
            for ti, v in enumerate(vals):
                taper_rows.append(
                    {
                        "timepoint": ti,
                        "taper_low": tl,
                        "Spectral_PCC_2D": v,
                    }
                )
        pd.DataFrame(taper_rows).to_csv(
            output_dir / "taper_sweep.csv",
            index=False,
        )
        log.info("Saved %s", output_dir / "taper_sweep.csv")

        # Taper plot: baselines + taper curves
        plot_taper_comparison(
            df,
            sweep_values,
            taper_sweep,
            output_dir / "taper_sweep_real.png",
            title=f"A549 Nuclei ({cfg.position}) — taper_low sweep",
        )
    else:
        log.warning("No metrics CSV at %s, skipping comparison plots.", metrics_csv)


if __name__ == "__main__":
    main()
