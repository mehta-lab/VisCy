"""Build multi-channel sample montages from saved per-channel crops.

Reads the single-cell crop PNGs written by ``cluster_cell_state.py`` under
``<output_dir>/<marker>/samples/<class>/<channel>/`` (channels: phase / organelle
/ sensor) and lays out one montage per class per marker in the style of the
witness / prob-sample figures: rows = **time-bin x channel**, columns = distinct
cells. Only the first channel row of each bin carries the bin label. Runs
independently of the clustering step so montages regenerate without re-running.

Usage
-----
python make_sample_montage.py -c zikv_infection_pop.yml
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from viscy_utils.cli_utils import load_config

# Class directory names, in (positive, negative) order; overridden from config.
DEFAULT_CLASSES = ("infected", "uninfected")
# Channel subdir order (top to bottom within a bin), matching cluster_cell_state.
CHANNEL_ORDER = ("phase", "organelle", "sensor")


def _bin_of(png: Path) -> tuple[int, str]:
    """Parse (bin index, bin label) from a ``bin{idx}_{label}_...`` filename."""
    parts = png.name.split("_", 2)
    return int(parts[0][3:]), parts[1]


def make_class_montage(class_dir: Path, out_path: Path, title: str) -> None:
    """Lay out one class as rows = (bin x channel), columns = distinct cells.

    Parameters
    ----------
    class_dir : Path
        ``samples/<class>/`` containing one subfolder per channel.
    out_path : Path
        Where to write the montage PNG.
    title : str
        Figure title (marker + class).
    """
    channels = [c for c in CHANNEL_ORDER if (class_dir / c).is_dir()]
    if not channels:
        raise FileNotFoundError(f"No channel subdirs under {class_dir}")

    # Index crops by (bin, cell filename) per channel; the filename (minus the
    # bin prefix) is the shared cell key across channels.
    ref = channels[0]
    cells: dict[int, list[str]] = {}
    bin_label: dict[int, str] = {}
    for p in sorted((class_dir / ref).glob("bin*_*.png")):
        b, label = _bin_of(p)
        bin_label[b] = label
        cells.setdefault(b, []).append(p.name)
    bins = sorted(cells)
    cols = max((len(v) for v in cells.values()), default=0)
    if cols == 0:
        raise FileNotFoundError(f"No bin-prefixed crops under {class_dir / ref}")

    n_rows = len(bins) * len(channels)
    fig, axes = plt.subplots(n_rows, cols, figsize=(cols * 1.4, n_rows * 1.5), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    for bi, b in enumerate(bins):
        for ci, ch in enumerate(channels):
            row = bi * len(channels) + ci
            label = f"{bin_label[b]}\n{ch}" if ci == 0 else ch
            axes[row][0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8)
            axes[row][0].axis("on")
            axes[row][0].set_xticks([])
            axes[row][0].set_yticks([])
            for c, fname in enumerate(cells[b]):
                fpath = class_dir / ch / fname
                if fpath.exists():
                    axes[row][c].imshow(mpimg.imread(fpath), cmap="gray")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    """Build a per-class multi-channel montage for every marker in the config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    classes = (cfg.get("positive_class", DEFAULT_CLASSES[0]), cfg.get("negative_class", DEFAULT_CLASSES[1]))
    out_root = Path(cfg["output_dir"])
    for entry in cfg["datasets"]:
        marker = entry["marker"]
        samples_dir = out_root / marker / "samples"
        for cls in classes:
            class_dir = samples_dir / cls
            if class_dir.is_dir():
                make_class_montage(
                    class_dir,
                    samples_dir / f"montage_{cls}.png",
                    f"{marker} — {cls}\nsamples by time bin x channel (phase / organelle / sensor)",
                )


if __name__ == "__main__":
    main()
