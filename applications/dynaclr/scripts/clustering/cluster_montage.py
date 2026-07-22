"""Build a top-K infection-enriched HDBSCAN-cluster montage from saved crops.

Reads the representative crop PNGs written by ``cluster_cell_state.py`` under
``<output_dir>/<marker>/cluster_samples/rank{r}_cluster{id}_inf{frac}/<channel>/``
and lays out one montage per marker: rows = **cluster x channel** (phase /
organelle / sensor stacked per cluster), columns = representative cells. Clusters
are already the top-K most infection-enriched (§5 well-condition scoring), ordered
by ``rank``. Runs independently of the clustering step (crops are on disk).

Usage
-----
python cluster_montage.py -c zikv_remodel_pop.yml
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from viscy_utils.cli_utils import load_config

# rank{r}_cluster{id}_inf{frac} — capture rank, cluster id, infected-well frac.
_CLUSTER_DIR = re.compile(r"^rank(\d+)_cluster(-?\d+)_inf([\d.]+)$")
CHANNEL_ORDER = ("phase", "organelle", "sensor")


def _cluster_dirs(cluster_root: Path) -> list[tuple[int, int, str, Path]]:
    """List (rank, cluster id, infected-frac label, dir), ordered by rank."""
    found = []
    for d in cluster_root.glob("rank*_cluster*"):
        m = _CLUSTER_DIR.match(d.name)
        if d.is_dir() and m:
            found.append((int(m.group(1)), int(m.group(2)), m.group(3), d))
    return sorted(found)


def make_cluster_montage(cluster_root: Path, out_path: Path, marker: str) -> None:
    """Tile crops as rows = (cluster x channel), columns = cells.

    Parameters
    ----------
    cluster_root : Path
        Directory containing ``rank{r}_cluster{id}_inf{frac}/<channel>/`` subdirs.
    out_path : Path
        Where to write the montage PNG.
    marker : str
        Marker name, used in the figure title.
    """
    clusters = _cluster_dirs(cluster_root)
    if not clusters:
        raise FileNotFoundError(f"No rank*_cluster*/ crop dirs under {cluster_root}")
    channels = [c for c in CHANNEL_ORDER if any((d / c).is_dir() for *_, d in clusters)]
    # Shared cell filenames per cluster come from the first channel present.
    cells = {}
    for rank, cid, frac, d in clusters:
        ref = next((c for c in channels if (d / c).is_dir()), None)
        cells[(rank, cid, frac)] = sorted(p.name for p in (d / ref).glob("*.png")) if ref else []
    cols = max((len(v) for v in cells.values()), default=0)
    if cols == 0:
        raise FileNotFoundError(f"No crops under {cluster_root}")

    n_rows = len(clusters) * len(channels)
    fig, axes = plt.subplots(n_rows, cols, figsize=(cols * 1.4, n_rows * 1.5), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    for ci_c, (rank, cid, frac, d) in enumerate(clusters):
        for chi, ch in enumerate(channels):
            row = ci_c * len(channels) + chi
            label = f"cluster {cid}\ninf={frac}\n{ch}" if chi == 0 else ch
            axes[row][0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8)
            axes[row][0].axis("on")
            axes[row][0].set_xticks([])
            axes[row][0].set_yticks([])
            for col, fname in enumerate(cells[(rank, cid, frac)]):
                fpath = d / ch / fname
                if fpath.exists():
                    axes[row][col].imshow(mpimg.imread(fpath), cmap="gray")

    fig.suptitle(
        f"{marker}\ntop-{len(clusters)} infection-enriched HDBSCAN clusters "
        f"(cluster x channel: phase / organelle / sensor)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    """Build a top-K cluster montage for every marker in the config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_root = Path(cfg["output_dir"])
    for entry in cfg["datasets"]:
        marker = entry["marker"]
        cluster_root = out_root / marker / "cluster_samples"
        make_cluster_montage(cluster_root, cluster_root / "montage_clusters.png", marker)


if __name__ == "__main__":
    main()
