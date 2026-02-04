"""WandB utilities for model registry."""

import tempfile
from pathlib import Path

import wandb

from viscy.airtable.schemas import ModelRecord


def download_model_artifact(
    model_name: str,
    version: str = "latest",
    wandb_project: str = "viscy-model-registry",
    download_dir: str | None = None,
) -> tuple[ModelRecord, Path]:
    """
    Download model artifact from WandB.

    Parameters
    ----------
    model_name : str
        Model name (e.g., 'contrastive-rpe1')
    version : str
        Version or alias (e.g., 'v1', 'latest', 'production')
    wandb_project : str
        WandB project name
    download_dir : str | None
        Where to download artifact files (temp dir if None)

    Returns
    -------
    tuple[ModelRecord, Path]
        (model_record, path_to_downloaded_config)
    """
    api = wandb.Api()
    entity = api.default_entity

    # Fetch artifact
    artifact = api.artifact(f"{entity}/{wandb_project}/{model_name}:{version}")

    # Parse metadata with Pydantic validation
    model_record = ModelRecord.from_wandb_artifact(artifact)

    # Download artifact files (config_fit.yml)
    if download_dir is None:
        download_dir = tempfile.mkdtemp(prefix=f"viscy_model_{model_name}_")

    artifact_dir = Path(artifact.download(root=download_dir))
    config_path = artifact_dir / "config_fit.yml"

    if not config_path.exists():
        raise FileNotFoundError(
            "Training config not found in artifact. "
            "Model may have been registered without config."
        )

    model_record.config_artifact_path = str(config_path)

    return model_record, config_path
