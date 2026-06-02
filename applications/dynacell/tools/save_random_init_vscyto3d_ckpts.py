"""Persist 4 random-init VSCyto3D checkpoints (one per organelle) for the Track A ablation.

Each Track A predict YAML is a separate CLI invocation, so without a frozen
checkpoint the 4 datasets per organelle would each draw a different random
init (different processes, different RNG state). One frozen ckpt per
organelle, shared across the organelle's 4 dataset predicts, is the only way
to keep the random-init baseline reproducible across iPSC + 3 A549 conditions.

Distinct seeds per organelle: each organelle's baseline is an independent
draw from the random-init distribution. Reproducibility is required *within*
an organelle (one ckpt shared across 4 dataset predicts), not *across*
organelles — using the same seed everywhere would test GT correlation with a
single random vector evaluated against 4 different GT channels.

Outputs (kept in lockstep with fcmae_vscyto3d_predict.yml arch hparams):
    /hpc/projects/comp.micro/virtual_staining/models/dynacell/randinit/<organelle>/fcmae_vscyto3d_pretrained/checkpoints/randinit.ckpt
"""

from __future__ import annotations

from pathlib import Path

import torch

from dynacell.engine import DynacellUNet

# Architecture hparams copied verbatim from configs/benchmarks/virtual_staining/
# _internal/shared/model/model_overlays/fcmae_vscyto3d_predict.yml.
# out_channels=1 because Track A predicts one channel per organelle
# (per-organelle dirs under <organelle>/fcmae_vscyto3d_pretrained/_no_train_randinit/).
MODEL_CONFIG: dict = {
    "in_channels": 1,
    "out_channels": 1,
    "encoder_blocks": [3, 3, 9, 3],
    "encoder_drop_path_rate": 0.1,
    "dims": [96, 192, 384, 768],
    "decoder_conv_blocks": 2,
    "stem_kernel_size": [5, 4, 4],
    "in_stack_depth": 15,
    "pretraining": False,
}

OUT_ROOT = Path("/hpc/projects/comp.micro/virtual_staining/models/dynacell/randinit")
ORGANELLES = ["nucl", "memb", "sec61b", "tomm20"]
BASE_SEED = 42


def main() -> None:
    """Save one frozen DynacellUNet random-init ckpt per organelle."""
    for idx, organelle in enumerate(ORGANELLES):
        seed = BASE_SEED + idx
        torch.manual_seed(seed)
        model = DynacellUNet(architecture="fcmae", model_config=MODEL_CONFIG)
        ckpt_dir = OUT_ROOT / organelle / "fcmae_vscyto3d_pretrained" / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        out = ckpt_dir / "randinit.ckpt"
        torch.save({"state_dict": model.state_dict()}, out)
        print(f"Wrote {out} (seed={seed})")


if __name__ == "__main__":
    main()
