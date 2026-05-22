"""Plot combined metrics from multiple positions on shared panels."""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation

OUTPUT_DIR = Path("output")

ALL_METRICS = [
    "PCC",
    "PSNR",
    "SSIM",
    "Spectral_PCC",
    "Spectral_PCC_OTF",
    "Spectral_PCC_Fixed",
    "Spectral_PCC_Wiener",
    "Spectral_PCC_SNR2",
    "Spectral_PCC_LogSNR",
    "Multiband_EV_NC",
    "Multiband_EV_PCC",
    "BL_PCC_DCR_XY",
    "BL_SSIM_DCR_XY",
    "BL_PCC_DCR_Z",
    "BL_SSIM_DCR_Z",
    "BL_PCC_FSC_XY",
    "BL_SSIM_FSC_XY",
    "BL_PCC_FSC_Z",
    "BL_SSIM_FSC_Z",
    "BL_PCC_OTF",
    "BL_SSIM_OTF",
    "FSC_XY",
    "FSC_Z",
    "FSC_GT_XY",
    "FSC_GT_Z",
    "DCR_XY",
    "DCR_Z",
    "DCR_A0",
    "DCR_r0",
    "PCC_2D",
    "PSNR_2D",
    "SSIM_2D",
    "Spectral_PCC_2D",
    "Spectral_PCC_Fixed_2D",
    "Spectral_PCC_Wiener_2D",
    "Spectral_PCC_SNR2_2D",
    "Spectral_PCC_LogSNR_2D",
    "Multiband_EV_NC_2D",
    "Multiband_EV_PCC_2D",
    "DCR_2D",
    "BL_PCC_DCR_2D",
    "BL_SSIM_DCR_2D",
    "BL_PCC_OTF_2D",
    "BL_SSIM_OTF_2D",
    "zero_frac",
]


def main():
    """Load per-position CSVs and plot median + MAD band for all metrics."""
    # Discover all position CSVs
    csv_files = sorted(OUTPUT_DIR.rglob("metrics.csv"))
    if not csv_files:
        print("No metrics.csv files found")
        sys.exit(1)

    # Load all positions
    positions: list[tuple[str, pd.DataFrame]] = []
    for csv_path in csv_files:
        pos_name = str(csv_path.parent.relative_to(OUTPUT_DIR))
        df = pd.read_csv(csv_path)
        positions.append((pos_name, df))

    print(f"Found {len(positions)} positions: {[p for p, _ in positions]}")

    # Determine which metrics are present
    all_cols = set()
    for _, df in positions:
        all_cols.update(df.columns)
    metrics = [m for m in ALL_METRICS if m in all_cols]
    n = len(metrics)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.asarray(axes).flatten()

    for i, name in enumerate(metrics):
        ax = axes[i]

        # Stack all positions into a matrix (positions x timepoints)
        all_series = []
        for _, pos_df in positions:
            if name in pos_df.columns:
                all_series.append(pos_df.set_index("timepoint")[name])
        if not all_series:
            ax.set_title(name, fontsize=10, fontweight="bold")
            ax.set_xlabel("Timepoint")
            ax.grid(True, alpha=0.3)
            continue

        combined = pd.concat(all_series, axis=1)
        t_vals = combined.index.values
        median_vals = combined.median(axis=1).values
        mad_vals = combined.apply(
            lambda row: median_abs_deviation(row.dropna()),
            axis=1,
        ).values

        # Median line
        ax.plot(t_vals, median_vals, color="C0", linewidth=1.5, label="median")
        # MAD band
        ax.fill_between(
            t_vals,
            median_vals - mad_vals,
            median_vals + mad_vals,
            alpha=0.25,
            color="C0",
            label="MAD",
        )

        # Linear fit on median
        finite = np.isfinite(median_vals)
        if finite.sum() > 1:
            slope, intercept = np.polyfit(t_vals[finite], median_vals[finite], 1)
            ax.plot(
                t_vals,
                slope * t_vals + intercept,
                color="red",
                linewidth=1,
                linestyle="--",
            )
            y0 = intercept
            yT = slope * t_vals[-1] + intercept
            drop = (y0 - yT) / y0 * 100 if y0 > 0 else 0
            cv = np.std(median_vals[finite]) / np.mean(median_vals[finite]) * 100
            ax.set_title(
                f"{name}\ndrop={drop:.1f}%  CV={cv:.1f}%",
                fontsize=10,
                fontweight="bold",
            )
        else:
            ax.set_title(name, fontsize=10, fontweight="bold")

        ax.set_xlabel("Timepoint")
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    n_pos = len(positions)
    fig.suptitle(
        f"A549 Nuclei — median +/- MAD across {n_pos} positions",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = OUTPUT_DIR / "combined_metrics.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_pcc_comparison():
    """Plot median PCC variants: 3D and 2D side by side."""
    csv_files = sorted(OUTPUT_DIR.rglob("metrics.csv"))
    if not csv_files:
        print("No metrics.csv files found")
        sys.exit(1)

    positions = []
    for csv_path in csv_files:
        positions.append(pd.read_csv(csv_path))

    # Matched colors across panels (same metric concept = same color)
    compare_3d = [
        ("PCC", "C3", "PCC"),
        ("BL_PCC_DCR_XY", "C0", "BL_PCC_DCR"),
        ("BL_PCC_FSC_XY", "C2", "BL_PCC_FSC"),
        ("BL_PCC_OTF", "C4", "BL_PCC_OTF"),
        ("Spectral_PCC", "C1", "Spectral_PCC"),
        ("Spectral_PCC_Fixed", "C5", "Spectral_PCC_Fixed"),
        ("Spectral_PCC_Wiener", "C6", "Spectral_PCC_Wiener"),
        ("Spectral_PCC_SNR2", "C7", "SNR^2"),
        ("Spectral_PCC_LogSNR", "C8", "LogSNR"),
        ("Multiband_EV_PCC", "tab:olive", "EV_PCC"),
    ]
    compare_2d = [
        ("PCC_2D", "C3", "PCC"),
        ("BL_PCC_DCR_2D", "C0", "BL_PCC_DCR"),
        ("BL_PCC_OTF_2D", "C4", "BL_PCC_OTF"),
        ("Spectral_PCC_2D", "C1", "Spectral_PCC"),
        ("Spectral_PCC_Fixed_2D", "C5", "Spectral_PCC_Fixed"),
        ("Spectral_PCC_Wiener_2D", "C6", "Spectral_PCC_Wiener"),
        ("Spectral_PCC_SNR2_2D", "C7", "SNR^2"),
        ("Spectral_PCC_LogSNR_2D", "C8", "LogSNR"),
        ("Multiband_EV_PCC_2D", "tab:olive", "EV_PCC"),
    ]

    fig, (ax3d, ax2d) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, variants, title in [
        (ax3d, compare_3d, "3D (full volume)"),
        (ax2d, compare_2d, "2D (mid-Z slice)"),
    ]:
        for col_name, color, label in variants:
            series = []
            for df in positions:
                if col_name in df.columns:
                    series.append(df.set_index("timepoint")[col_name])
            if not series:
                continue
            combined = pd.concat(series, axis=1)
            t = combined.index.values
            med = combined.median(axis=1).values
            ax.plot(t, med, color=color, linewidth=2, label=label)

        ax.set_xlabel("Timepoint", fontsize=12)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    ax3d.set_ylabel("PCC", fontsize=12)

    n_pos = len(positions)
    fig.suptitle(
        f"A549 Nuclei — median across {n_pos} positions",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = OUTPUT_DIR / "pcc_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def print_weighted_summary():
    """Print per-position weighted summary using DCR_w reliability weights."""
    csv_files = sorted(OUTPUT_DIR.rglob("metrics.csv"))
    if not csv_files:
        print("No metrics.csv files found")
        return

    positions = []
    for csv_path in csv_files:
        pos_name = str(csv_path.parent.relative_to(OUTPUT_DIR))
        df = pd.read_csv(csv_path)
        positions.append((pos_name, df))

    # Metrics to summarize
    summary_metrics = [
        "PCC",
        "Spectral_PCC",
        "Spectral_PCC_SNR2",
        "Spectral_PCC_LogSNR",
        "Multiband_EV_PCC",
        "BL_PCC_DCR_XY",
    ]

    has_weights = any("DCR_w" in df.columns for _, df in positions)
    if not has_weights:
        print("No DCR_w column found — skipping weighted summary")
        return

    header = f"{'Metric':30s} {'CV%':>6s} {'Drop%':>6s}"
    header += f" {'CV_w%':>6s} {'Drop_w%':>7s} {'Scor%':>6s}"
    print("\n=== Weighted summary (per-position, then median) ===")
    print(header)

    for col in summary_metrics:
        # Per-position stats
        drops_uw, drops_w, cvs_uw, cvs_w, scorables = [], [], [], [], []
        for _, df in positions:
            if col not in df.columns or "DCR_w" not in df.columns:
                continue
            t = df["timepoint"].values
            vals = df[col].values
            w = df["DCR_w"].values
            finite = np.isfinite(vals) & np.isfinite(w)
            if finite.sum() < 2:
                continue

            v, ww, tt = vals[finite], w[finite], t[finite]

            # Unweighted drop (stable formula)
            slope, intercept = np.polyfit(tt, v, 1)
            y0 = intercept + slope * tt[0]
            yT = intercept + slope * tt[-1]
            drop_uw = (y0 - yT) / y0 * 100 if y0 > 0 else 0
            drops_uw.append(drop_uw)

            # Unweighted CV
            cvs_uw.append(np.std(v) / np.mean(v) * 100 if np.mean(v) != 0 else 0)

            # Weighted drop
            w_sum = ww.sum()
            if w_sum > 0:
                slope_w, intercept_w = np.polyfit(tt, v, 1, w=ww)
                y0_w = intercept_w + slope_w * tt[0]
                yT_w = intercept_w + slope_w * tt[-1]
                drop_w = (y0_w - yT_w) / y0_w * 100 if y0_w > 0 else 0
                drops_w.append(drop_w)

                # Weighted CV
                mu_w = np.average(v, weights=ww)
                var_w = np.average((v - mu_w) ** 2, weights=ww)
                cv_w = np.sqrt(var_w) / mu_w * 100 if mu_w != 0 else 0
                cvs_w.append(cv_w)

                scorables.append(np.mean(ww))
            else:
                drops_w.append(np.nan)
                cvs_w.append(np.nan)
                scorables.append(0.0)

        if not drops_uw:
            continue

        cv_med = np.nanmedian(cvs_uw)
        drop_med = np.nanmedian(drops_uw)
        cv_w_med = np.nanmedian(cvs_w)
        drop_w_med = np.nanmedian(drops_w)
        scor_med = np.nanmedian(scorables) * 100

        line = f"{col:30s} {cv_med:6.1f} {drop_med:6.1f}"
        line += f" {cv_w_med:6.1f} {drop_w_med:7.1f} {scor_med:6.1f}"
        print(line)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.output_dir is not None:
        global OUTPUT_DIR  # noqa: PLW0603
        OUTPUT_DIR = args.output_dir
    main()
    plot_pcc_comparison()
    print_weighted_summary()
