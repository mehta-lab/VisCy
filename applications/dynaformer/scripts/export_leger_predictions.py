"""Export whole-track dynaformer predictions in the Leger baseline harness's format.

Writes ``<out>/<run>_<split>.npz`` with the same schema
``Andrea_mlbi/code/predict.py`` produces (``names``, ``pred_<name>``, ``true_<name>``), so the
verified metrics in ``Andrea_mlbi/code/reproduce/`` can score dynaformer alongside the released
checkpoints without any per-model special casing.

Whole tracks are passed in one forward at batch size 1 -- no sub-sampling, no truncation. RoPE
imposes no length limit, so this is exact rather than an approximation.

Usage
-----
```
python export_leger_predictions.py --ckpt <path>.ckpt --run dynaformer_temporal --split test
```
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from jsonargparse import ArgumentParser

from dynaformer.models import SpatioTemporalModel
from viscy_data.npy_track import NpyTrackDataset, _track_names
from viscy_utils.compose import load_composed_config

DEFAULT_DATA = Path("/hpc/projects/comp.micro/virtual_staining/models/dynaformer/data")
DEFAULT_OUT = Path("/hpc/projects/comp.micro/virtual_staining/models/dynaformer/Andrea_mlbi/code/results/predictions")


def build_model(config: Path, ckpt: Path) -> SpatioTemporalModel:
    """Rebuild the network from its training config and load the checkpoint weights.

    ``DynaformerModule.load_from_checkpoint`` cannot be used: LightningCLI keeps the model
    specification in a sidecar ``config.yaml`` rather than in the checkpoint's
    ``hyper_parameters``, so the saved hyperparameters alone do not describe the architecture.
    Composing the training config is the reliable route, and it keeps this script honest about
    which run produced the weights.
    """
    composed = load_composed_config(config)
    parser = ArgumentParser(exit_on_error=False)
    parser.add_argument("--model", type=SpatioTemporalModel)
    parsed = parser.parse_object({"model": composed["model"]["init_args"]["model"]})
    # ``instantiate_classes`` is the pre-4.49 spelling; keep working on either.
    instantiate = getattr(parser, "instantiate", None) or parser.instantiate_classes
    model = instantiate(parsed).model

    state = torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict({k.removeprefix("model."): v for k, v in state.items()}, strict=True)
    return model.eval()


@torch.inference_mode()
def export(
    ckpt: Path, config: Path, run: str, split: str, data_path: Path, out_dir: Path, device: str = "cuda"
) -> Path:
    """Predict every track of a split and save predictions with their ground truth."""
    model = build_model(config, ckpt).to(device)

    root = data_path / split
    dataset = NpyTrackDataset(root, _track_names(root), source_channel=0, whole_track=True)

    predictions, truths = {}, {}
    for index in range(len(dataset)):
        batch = dataset[index]
        _, heads = model(batch["source"].to(device))
        name = batch["index"][0]
        predictions[name] = heads["scalar"][0].float().cpu().numpy()
        truths[name] = batch["scalar_target"][0].numpy()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run}_{split}.npz"
    np.savez_compressed(
        path,
        names=np.array(list(predictions)),
        **{f"pred_{n}": p for n, p in predictions.items()},
        **{f"true_{n}": t for n, t in truths.items()},
    )
    errors = [np.abs(predictions[n] - truths[n]).mean() for n in predictions]
    print(f"{run} / {split}: {len(predictions)} tracks, track L1 = {np.mean(errors):.6f}  ->  {path}")
    return path


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="the fit leaf YAML this run used")
    parser.add_argument("--run", required=True, help="row label in the comparison table")
    parser.add_argument("--split", nargs="+", default=["test", "test_drugged"])
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    for split in args.split:
        export(args.ckpt, args.config, args.run, split, args.data_path, args.out_dir, args.device)


if __name__ == "__main__":
    main()
