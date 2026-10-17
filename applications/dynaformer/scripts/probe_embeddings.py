"""Linear probe of each cached foundation-model embedding: how much of the reporters a single frame carries.

Usage
-----
python probe_embeddings.py [--data-dir <Bill/5-cellcycle>] [--alphas 1 10 100 1000]

For every encoder cache in ``<data-dir>/embeddings``, fits a ridge regression from one frame's embedding (identity
view) to its normalized Geminin-mKO and PIP-mVenus medians on the training split, picks the penalty on the
validation split and reports Pearson r on the test split, for the global token, the central patch-token mean, and
both. No temporal context: this is the per-frame information the temporal model starts from.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = ["mko", "mvenus"]


def ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge weights (with an unpenalized intercept) of standardized features."""
    gram = x.T @ x + alpha * np.eye(x.shape[1])
    return np.linalg.solve(gram, x.T @ y)


def pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Column-wise Pearson r."""
    a, b = a - a.mean(0), b - b.mean(0)
    return (a * b).sum(0) / np.sqrt((a**2).sum(0) * (b**2).sum(0))


def main() -> None:
    """Probe every encoder cache and print a table."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("/hpc/projects/virtual_staining/datasets/huang-lab/Bill/5-cellcycle")
    )
    parser.add_argument("--alphas", type=float, nargs="+", default=[1, 10, 100, 1000, 10000])
    args = parser.parse_args()
    points = pd.read_csv(args.data_dir / "points.csv", usecols=["row", "track_id", *TARGETS])
    split = pd.read_csv(args.data_dir / "split_tracks.csv", usecols=["track_id", "split"])
    points = points.merge(split, on="track_id", how="left", validate="many_to_one")
    scale = np.asarray(json.loads((args.data_dir / "stats.json").read_text())["fucci"]["scale"])
    y = points[TARGETS].to_numpy() / scale
    rows = {name: points.row[points.split == name].to_numpy() for name in ("train", "val", "test")}

    results = []
    for cache in sorted((args.data_dir / "embeddings").glob("*.npy")):
        meta = json.loads(cache.with_suffix(".json").read_text())
        embeddings = np.load(cache, mmap_mode="r")
        features = np.asarray(embeddings[:, 0], dtype=np.float32)  # identity view
        token = meta["token_dim"]
        for part, columns in (("token", slice(0, token)), ("center", slice(token, None)), ("all", slice(None))):
            x = features[:, columns]
            mean, std = x[rows["train"]].mean(0), x[rows["train"]].std(0) + 1e-6
            x = (x - mean) / std
            y_mean = y[rows["train"]].mean(0)
            best = None
            for alpha in args.alphas:
                weights = ridge(x[rows["train"]], y[rows["train"]] - y_mean, alpha)
                val_r = pearson(x[rows["val"]] @ weights + y_mean, y[rows["val"]]).mean()
                if best is None or val_r > best[0]:
                    best = (val_r, alpha, weights)
            _, alpha, weights = best
            test_r = pearson(x[rows["test"]] @ weights + y_mean, y[rows["test"]])
            results.append(
                {
                    "encoder": meta["encoder"],
                    "features": part,
                    "dims": x.shape[1],
                    "alpha": alpha,
                    "r mKO": test_r[0],
                    "r mVenus": test_r[1],
                    "r mean": test_r.mean(),
                }
            )
            print(results[-1], flush=True)
    print(pd.DataFrame(results).to_markdown(index=False, floatfmt=".3f"))


if __name__ == "__main__":
    main()
