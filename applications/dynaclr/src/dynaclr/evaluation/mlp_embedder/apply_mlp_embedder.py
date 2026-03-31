"""CLI for applying a trained MLP embedder to extract penultimate-layer representations.

Usage:
    dynaclr apply-mlp-embedder -c path/to/config.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import anndata as ad
import click
import numpy as np
import torch
from pydantic import BaseModel, Field

from viscy_models.components.heads import MLP
from viscy_utils.cli_utils import load_config
from viscy_utils.evaluation.zarr_utils import append_to_anndata_zarr
from viscy_utils.tensor_utils import to_numpy


class MlpEmbedderApplyConfig(BaseModel):
    """Configuration for MLP embedder inference.

    Parameters
    ----------
    embeddings_path : str
        Path to the AnnData zarr store with embeddings in ``.X``.
    model_path : str
        Path to the ``.pt`` checkpoint saved by train-mlp-embedder.
    output_path : Optional[str]
        Path to write output zarr. When ``None``, writes back to ``embeddings_path``.
    batch_size : int
        Inference batch size.
    """

    embeddings_path: str = Field(..., min_length=1)
    model_path: str = Field(..., min_length=1)
    output_path: Optional[str] = Field(default=None)
    batch_size: int = Field(default=256, gt=0)


def _load_model(model_path: Path, device: torch.device) -> MLP:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = MLP(
        in_dims=checkpoint["input_dim"],
        hidden_dims=checkpoint["hidden_dims"],
        dropout=checkpoint["dropout"],
        norm="bn" if checkpoint["batch_norm"] else "ln",
        num_classes=checkpoint["num_classes"],
        cosine_classifier=checkpoint["cosine_classifier"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file",
)
def main(config: Path) -> None:
    """Apply a trained MLP embedder to extract penultimate-layer representations."""
    click.echo("=" * 60)
    click.echo("MLP EMBEDDER INFERENCE")
    click.echo("=" * 60)

    config_dict = load_config(config)
    cfg = MlpEmbedderApplyConfig(**config_dict)

    write_path = Path(cfg.output_path) if cfg.output_path is not None else Path(cfg.embeddings_path)

    click.echo(f"\n  Embeddings: {cfg.embeddings_path}")
    click.echo(f"  Model: {cfg.model_path}")
    click.echo(f"  Output: {write_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    click.echo(f"  Device: {device}")

    adata = ad.read_zarr(cfg.embeddings_path)
    click.echo(f"\n  Loaded embeddings: {adata.shape}")

    model = _load_model(Path(cfg.model_path), device)
    click.echo(f"  Loaded model: input_dim={model.input_dim}")

    X = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
    X_t = torch.tensor(X, dtype=torch.float32)

    reps: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(X_t), cfg.batch_size):
            batch = X_t[i : i + cfg.batch_size].to(device)
            reps.append(model.encode(batch))

    X_mlp = to_numpy(torch.cat(reps, dim=0))
    click.echo(f"  Extracted representations: {X_mlp.shape}")

    append_to_anndata_zarr(write_path, obsm={"X_mlp": X_mlp})
    click.echo(f"  Wrote .obsm['X_mlp'] to {write_path}")
    click.echo("Done.")


if __name__ == "__main__":
    main()
