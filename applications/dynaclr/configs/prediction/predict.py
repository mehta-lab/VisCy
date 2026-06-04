"""DynaCLR inference script.

Load a trained ContrastiveModule checkpoint and run inference on
pre-normalized tensors. Edit the paths and parameters below before running.

Usage
-----
    uv run python predict.py

Input tensors must be shape (N, C, Z, Y, X) and already normalized.
"""

import torch

from dynaclr.engine import ContrastiveModule
from viscy_models.contrastive import ContrastiveEncoder


def main():
    """Run DynaCLR inference on pre-normalized tensors."""
    # ── paths ─────────────────────────────────────────────────────────────────
    checkpoint = (
        "/hpc/projects/organelle_phenotyping/models/SEC61_TOMM20_G3BP1_Sensor/"
        "time_interval/dynaclr_gfp_rfp_Ph/organelle_sensor_phase_maxproj_ver3_150epochs/"
        "saved_checkpoints/epoch=104-step=53760.ckpt"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64

    # ── encoder config (must match training) ──────────────────────────────────
    encoder_config = {
        "backbone": "convnext_tiny",
        "in_channels": 1,
        "in_stack_depth": 1,
        "stem_kernel_size": (1, 4, 4),
        "stem_stride": (1, 4, 4),
        "embedding_dim": 768,
        "projection_dim": 32,
        "drop_path_rate": 0.0,
    }

    # ── load model ────────────────────────────────────────────────────────────
    encoder = ContrastiveEncoder(**encoder_config)
    model = ContrastiveModule.load_from_checkpoint(checkpoint, map_location=device, encoder=encoder)
    model.eval()
    model.to(device)

    # ── run inference ─────────────────────────────────────────────────────────
    # TODO load tensors . Model expect normalization 0-1 as input and was trained with (1,160,160) voxels
    tensors = torch.randn(100, 2, 30, 1024, 1024).to(device)

    all_features = []
    # all_projections = [] # TODO uncomment  the projections if you want to use them

    with torch.inference_mode():
        for i in range(0, len(tensors), batch_size):
            batch = tensors[i : i + batch_size].to(device)
            features, projections = model(batch)
            all_features.append(features.cpu())
            # all_projections.append(projections.cpu())

    embeddings = {
        "features": torch.cat(all_features, dim=0),  # (N, embedding_dim)
        # "projections": torch.cat(all_projections, dim=0),  # (N, projection_dim)
    }

    print(embeddings.shape)


if __name__ == "__main__":
    main()
