"""Generate the shading analysis comparison figure.

Reads metrics CSVs from four simulation conditions and produces a 2x2
plot showing how nbins_low fixes the shading artifact across metric variants.

Usage::

    uv run python -m dynacell.evaluation.spectral_pcc.plot_shading_analysis
    uv run python -m dynacell.evaluation.spectral_pcc.plot_shading_analysis --root-dir /path/to/outputs
"""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Paths (CWD-relative defaults, overridable via CLI) ---
ROOT = Path(".")
OUT = ROOT / "output_sim_shading"

CSVS = {
    "noshade_nofix": OUT / "simulation_metrics_noshade_nofix.csv",
    "shade_nofix": OUT / "simulation_metrics_shade_nofix.csv",
    "noshade_fix": ROOT / "output_simulation" / "simulation_metrics.csv",
    "shade_fix": OUT / "simulation_metrics.csv",
}


def load():
    """Load simulation metric CSVs into a dict keyed by run name."""
    dfs = {}
    for name, path in CSVS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Re-run simulations first.")
        dfs[name] = pd.read_csv(path)
    return dfs


def main():
    """Generate PCC comparison plots for shading vs no-shading simulations."""
    dfs = load()
    t = dfs["noshade_nofix"]["timepoint"].values

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    fig.suptitle(
        "Effect of illumination shading (beta=0.01) on metrics",
        fontsize=14,
        fontweight="bold",
    )

    # Color scheme
    C_NOSHADE = "#2176AE"
    C_NOFIX = "#D7263D"
    C_FIX = "#1B998B"
    LW = 1.8

    three_cond = [
        ("noshade_nofix", "No shading", C_NOSHADE, "-"),
        ("shade_nofix", "Shading, nbins_low=0", C_NOFIX, "--"),
        ("shade_fix", "Shading, nbins_low=3", C_FIX, "-"),
    ]

    # --- (0,0) PCC — 2 conditions only (nbins_low irrelevant) ---
    ax = axes[0, 0]
    ax.plot(
        t,
        dfs["noshade_nofix"]["PCC_2D"],
        color=C_NOSHADE,
        ls="-",
        lw=LW,
        label="No shading",
    )
    ax.plot(
        t,
        dfs["shade_fix"]["PCC_2D"],
        color=C_NOFIX,
        ls="--",
        lw=LW,
        label="With shading",
    )
    ax.set_title("PCC (no frequency filtering)", fontsize=12)
    ax.set_xlabel("Timepoint")
    ax.set_ylabel("PCC")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc="lower left")

    # --- (0,1) Spectral_PCC — 3 conditions ---
    ax = axes[0, 1]
    for dfkey, label, color, ls in three_cond:
        ax.plot(t, dfs[dfkey]["Spectral_PCC_2D"], color=color, ls=ls, lw=LW, label=label)
    ax.set_title("Spectral_PCC", fontsize=12)
    ax.set_xlabel("Timepoint")
    ax.set_ylabel("PCC")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc="lower left")

    # --- (1,0) DCR — 2 conditions ---
    ax = axes[1, 0]
    ax.plot(
        t,
        dfs["noshade_nofix"]["DCR_2D"],
        color=C_NOSHADE,
        ls="-",
        lw=LW,
        label="No shading",
    )
    ax.plot(
        t,
        dfs["shade_fix"]["DCR_2D"],
        color=C_NOFIX,
        ls="--",
        lw=LW,
        label="With shading",
    )
    ax.set_title("DCR resolution", fontsize=12)
    ax.set_xlabel("Timepoint")
    ax.set_ylabel("Resolution (um)")
    ax.legend(fontsize=8, loc="best")

    # --- (1,1) FRC cutoff ---
    ax = axes[1, 1]
    if "BL_PCC_DCR_2D" in dfs["noshade_nofix"].columns:
        ax.plot(
            t,
            dfs["noshade_nofix"]["BL_PCC_DCR_2D"],
            color=C_NOSHADE,
            ls="-",
            lw=LW,
            label="No shading",
        )
        ax.plot(
            t,
            dfs["shade_nofix"]["BL_PCC_DCR_2D"],
            color=C_NOFIX,
            ls="--",
            lw=LW,
            label="Shading, nbins_low=0",
        )
        ax.plot(
            t,
            dfs["shade_fix"]["BL_PCC_DCR_2D"],
            color=C_FIX,
            ls="-",
            lw=LW,
            label="Shading, nbins_low=3",
        )
        ax.set_title("BL_PCC (DCR cutoff)", fontsize=12)
        ax.set_xlabel("Timepoint")
        ax.set_ylabel("PCC")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8, loc="lower left")
    else:
        ax.set_visible(False)

    outpath = OUT / "shading_comparison.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpath}")


def _rebuild_paths(root: Path) -> None:
    """Rebuild module-level ROOT, OUT, and CSVS from a new root directory."""
    global ROOT, OUT, CSVS
    ROOT = root
    OUT = ROOT / "output_sim_shading"
    CSVS = {
        "noshade_nofix": OUT / "simulation_metrics_noshade_nofix.csv",
        "shade_nofix": OUT / "simulation_metrics_shade_nofix.csv",
        "noshade_fix": ROOT / "output_simulation" / "simulation_metrics.csv",
        "shade_fix": OUT / "simulation_metrics.csv",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.root_dir is not None:
        _rebuild_paths(args.root_dir)
    main()
