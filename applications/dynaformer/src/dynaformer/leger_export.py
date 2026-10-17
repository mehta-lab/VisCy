"""Export a Leger cell-cycle run's whole-track predictions in the Leger baseline harness's format.

Usage
-----
python -m dynaformer.leger_export --run-dir <runs>/cellcycle_leger/regression_c-radio --leaf regression_c-radio.yml

Reads ``<run-dir>/eval/<checkpoint>/predictions_<split>.csv`` (written by ``run_eval.slurm``) and writes
``<out>/<name>_<split>.npz`` with ``names``, ``pred_<track>`` and ``true_<track>``, each ``(T, 2)`` in stored label
order (Geminin, Cdt1): the schema of ``Andrea_mlbi/code/predict.py``, so ``reproduce.report`` scores the run with the
same metrics as the released checkpoints. The truths written are the stored labels themselves; the CSV's copies are
checked against them, which also checks that every track of the split is there, whole and in time order.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dynaformer.leger_data import REPORTERS, load_index

ROOT = Path("/hpc/projects/comp.micro/virtual_staining/models/dynaformer")
DEFAULT_DATA = ROOT / "data"
DEFAULT_OUT = ROOT / "Andrea_mlbi" / "code" / "results" / "predictions"
#: Largest difference allowed between a CSV truth and its stored label: the CSV keeps 4 significant digits.
CSV_TOLERANCE = 1e-4


def run_name(leaf: str) -> str:
    """Row label of a leaf in the harness tables: ``regression_<encoder>[_single_frame]`` -> ``dynaformer_...``."""
    stem = Path(leaf).stem
    if not stem.startswith("regression_"):
        raise ValueError(f"Only regression leaves have a harness row, not {leaf!r}.")
    encoder = stem.removeprefix("regression_")
    if encoder.endswith("_single_frame"):
        return f"dynaformer_{encoder.removesuffix('_single_frame')}_single_frame"
    return f"dynaformer_{encoder}_temporal"


def export(
    run_dir: Path,
    name: str,
    split: str,
    data_dir: Path = DEFAULT_DATA,
    index_dir: Path | None = None,
    out_dir: Path = DEFAULT_OUT,
    checkpoint: str = "best_regression",
) -> Path:
    """Write one split's predictions of one run as ``<out_dir>/<name>_<split>.npz``.

    Raises
    ------
    ValueError
        If the CSV is missing a track of the split, holds frames of another, or its truths differ from the labels.
    """
    index, _ = load_index(index_dir or data_dir / "embeddings")
    frames = pd.read_csv(run_dir / "eval" / checkpoint / f"predictions_{split}.csv")
    tracks = index[index.split == split].groupby("track_index", sort=True).agg(name=("name", "first"), n=("t", "size"))
    if set(frames.track_id) != set(tracks.index):
        raise ValueError(f"{run_dir} {split}: the predicted tracks are not the tracks of the split.")
    predictions, truths = {}, {}
    for track_index, track in frames.groupby("track_id", sort=True):
        track = track.sort_values("t")
        track_name = tracks.loc[track_index, "name"]
        if not np.array_equal(track.t.to_numpy(), np.arange(tracks.loc[track_index, "n"])):
            raise ValueError(f"{split}/{track_name}: predictions do not cover the track's frames once each.")
        stored = np.load(data_dir / split / "labels" / f"{track_name}.npy")[:, :, 0].T.astype(np.float32)
        written = track[[f"true_{r}" for r in REPORTERS]].to_numpy(np.float32)
        if np.abs(written - stored).max() > CSV_TOLERANCE:
            raise ValueError(f"{split}/{track_name}: the CSV's truths are not the stored labels.")
        predictions[track_name] = track[[f"regression_{r}" for r in REPORTERS]].to_numpy(np.float32)
        truths[track_name] = stored
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}_{split}.npz"
    names = sorted(predictions)
    np.savez_compressed(
        path,
        names=np.array(names),
        **{f"pred_{n}": predictions[n] for n in names},
        **{f"true_{n}": truths[n] for n in names},
    )
    errors = [np.abs(predictions[n] - truths[n]).mean() for n in names]
    print(f"{name} / {split}: {len(names)} tracks, track L1 = {np.mean(errors):.6f}  ->  {path}")
    return path


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--leaf", required=True, help="the experiment leaf the run trained, for its row label")
    parser.add_argument("--split", nargs="+", default=["test", "test_drugged"])
    parser.add_argument("--checkpoint", default="best_regression")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    for split in args.split:
        export(
            args.run_dir, run_name(args.leaf), split, args.data_dir, out_dir=args.out_dir, checkpoint=args.checkpoint
        )


if __name__ == "__main__":
    main()
