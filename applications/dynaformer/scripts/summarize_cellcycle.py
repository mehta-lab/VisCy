"""Collect the scores of every cell-cycle run into one table.

Usage
-----
python summarize_cellcycle.py [--runs <root>/runs/cellcycle] [--split test]

Reads ``<runs>/<run>/eval/<checkpoint>/metrics_<split>.json`` (written by ``run_eval.slurm``) and keeps, for each run
and readout, the checkpoint selected on that readout: a multitask run contributes its regression readout from
``best_regression`` and its stain readout from ``best_stain``. Writes ``<runs>/summary_<split>.csv`` and prints a
Markdown table.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

COLUMNS = {
    "pearson_mko": "r mKO",
    "pearson_mvenus": "r mVenus",
    "r2": "R²",
    "mae": "MAE",
    "track_pearson": "track r",
    "phase_acc": "phase acc",
    "phase_f1": "phase F1",
    "pixel_pearson": "pixel r",
}


def collect(runs: Path, split: str) -> pd.DataFrame:
    """One row per (run, readout), from the checkpoint selected on that readout."""
    rows = []
    for path in sorted(runs.glob(f"*/eval/best_*/metrics_{split}.json")):
        run, checkpoint = path.parts[-4], path.parts[-2]
        task, _, encoder = run.partition("_")
        for readout, metrics in json.loads(path.read_text()).items():
            if checkpoint != f"best_{readout}":
                continue
            rows.append({"run": run, "task": task, "encoder": encoder or "-", "readout": readout, **metrics})
    return pd.DataFrame(rows)


def main() -> None:
    """Print and save the summary table."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--runs", type=Path, default=Path("/hpc/projects/comp.micro/virtual_staining/models/dynaformer/runs/cellcycle")
    )
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()
    table = collect(args.runs, args.split)
    if table.empty:
        raise SystemExit(f"No metrics_{args.split}.json under {args.runs}.")
    table = table.sort_values(["readout", "task", "encoder"])
    table.to_csv(args.runs / f"summary_{args.split}.csv", index=False)
    shown = table[["readout", "run"] + [c for c in COLUMNS if c in table]].rename(columns=COLUMNS)
    print(shown.to_markdown(index=False, floatfmt=".3f"))


if __name__ == "__main__":
    main()
