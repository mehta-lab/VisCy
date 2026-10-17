"""Render virtual-staining examples of trained cell-cycle runs along test tracks.

Usage
-----
python render_stain_examples.py --runs stain_unconditioned stain_c-radio --out <dir> [--tracks 3 --frames 6]

Rebuilds each run from its resolved ``config.yaml`` and ``checkpoints/best_stain.ckpt``, runs whole test tracks (the
longest that pass through S phase) and writes ``stain_track<id>.png``: per frame, the phase input, the true
Geminin-mKO and PIP-mVenus, and every run's prediction of both. All runs share the crops, so the panels line up.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from lightning.pytorch.cli import LightningCLI

from dynaformer.cellcycle_data import CellCycleDataModule
from dynaformer.cellcycle_engine import TemporalEmbeddingModule
from dynaformer.cellcycle_metrics import phase_strict

RUNS = Path("/hpc/projects/comp.micro/virtual_staining/models/dynaformer/runs/cellcycle")


def load_run(run: str) -> tuple[TemporalEmbeddingModule, CellCycleDataModule]:
    """Instantiate a run's model and data module from its config and load its best stain checkpoint."""
    # The saved fit config also holds fit-only keys (ckpt_path, ...); only the model and the data are rebuilt.
    config = yaml.safe_load((RUNS / run / "config.yaml").read_text())
    cli = LightningCLI(
        TemporalEmbeddingModule,
        CellCycleDataModule,
        run=False,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=None,
        args={"model": config["model"], "data": config["data"], "trainer": {"logger": False}},
    )
    state = torch.load(RUNS / run / "checkpoints" / "best_stain.ckpt", map_location="cpu", weights_only=False)
    cli.model.load_state_dict(state["state_dict"])
    return cli.model.eval(), cli.datamodule


@torch.no_grad()
def predict_track(module: TemporalEmbeddingModule, dataset, clip: int, device: torch.device) -> dict:
    """Run one whole track, every frame imaged."""
    batch = dataset.__getitems__([clip])
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = module(batch)
    return {"batch": batch, "stain": out["stain"].float()}


def main() -> None:
    """Render the example grids."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--tracks", type=int, default=3)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    loaded = {}
    for run in args.runs:
        module, datamodule = load_run(run)
        datamodule.setup("test")
        dataset = datamodule.test_dataset
        dataset.image_stride = 1  # every frame imaged
        loaded[run] = (module.to(device), dataset)
    dataset = next(iter(loaded.values()))[1]
    thresholds = json.loads((dataset.data_dir / "stats.json").read_text())["thresholds"]
    scale = dataset.scale

    # The longest test tracks that pass through S phase.
    candidates = []
    for clip, (start, length, _) in enumerate(dataset.clips):
        counts = dataset._targets[start : start + length] * scale
        phase = phase_strict(counts[:, 0], counts[:, 1], thresholds)
        if (phase == 1).sum() >= 5:
            candidates.append((length, clip))
    chosen = [clip for _, clip in sorted(candidates, reverse=True)[: args.tracks]]

    for clip in chosen:
        results = {run: predict_track(module, ds, clip, device) for run, (module, ds) in loaded.items()}
        batch = next(iter(results.values()))["batch"]
        frames = batch["rows"].shape[1]
        picks = np.linspace(0, frames - 1, args.frames).round().astype(int)
        columns = ["phase", "true mKO", "true mVenus"] + [f"{run}\n{c}" for run in args.runs for c in ("mKO", "mVenus")]
        fig, axes = plt.subplots(len(picks), len(columns), figsize=(1.9 * len(columns), 1.9 * len(picks)))
        for i, frame in enumerate(picks):
            target = batch["target_image"][frame].cpu().numpy()
            images = [batch["source"][frame, 0].cpu().numpy(), target[0], target[1]]
            for run in args.runs:
                images += list(results[run]["stain"][frame].cpu().numpy())
            true_levels = batch["target"][0, frame].cpu().numpy() * scale
            for j, (image, title) in enumerate(zip(images, columns, strict=True)):
                ax = axes[i, j]
                if j == 0:
                    ax.imshow(image, cmap="gray", vmin=-3, vmax=6)
                    ax.set_ylabel(
                        f"t={int(batch['t'][0, frame])}\nmKO {true_levels[0]:.0f}\nmVenus {true_levels[1]:.0f}",
                        fontsize=7,
                    )
                else:
                    reporter = 0 if "mKO" in title else 1
                    ax.imshow(image, cmap="magma" if reporter == 0 else "viridis", vmin=0, vmax=1.0)
                if i == 0:
                    ax.set_title(title, fontsize=7)
                ax.set_xticks([])
                ax.set_yticks([])
        track_id = int(batch["track_id"][0])
        fig.suptitle(f"test track {track_id}: virtual staining of the reporters from phase", fontsize=9)
        fig.tight_layout()
        fig.savefig(args.out / f"stain_track{track_id}.png", dpi=100)
        plt.close(fig)
        print(f"Wrote {args.out / f'stain_track{track_id}.png'}")


if __name__ == "__main__":
    main()
