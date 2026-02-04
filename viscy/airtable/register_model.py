"""Register trained models to W&B artifact registry and shared directory."""

import shutil
from pathlib import Path
from typing import Any

import wandb


def register_model(
    checkpoint_path: str,
    model_name: str,
    model_type: str,
    version: str,
    config_path: str | None = None,
    aliases: list[str] | None = None,
    wandb_run_id: str | None = None,
    wandb_project: str = "viscy-model-registry",
    shared_dir: str = "/hpc/models/shared",
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
    airtable_base_id: str | None = None,
    airtable_collection_id: str | None = None,
) -> str:
    """
    Register a trained model to W&B artifacts and copy to shared directory.

    This creates a W&B artifact with references (not uploads) to track model lineage.
    The checkpoint file is copied to a shared HPC directory for team access.

    Parameters
    ----------
    checkpoint_path : str
        Path to Lightning checkpoint (.ckpt file)
    model_name : str
        Human-readable name (e.g., "contrastive-rpe1")
    model_type : str
        Model category: contrastive, segmentation, vae, translation
    version : str
        Semantic version (e.g., "v1", "v2", "v3")
    config_path : str | None
        Path to training config YAML file (config_fit.yml). This will be stored
        in the WandB artifact for later use in prediction.
    aliases : list[str] | None
        Tags like "production", "best", "latest"
    wandb_run_id : str | None
        W&B run ID that trained this model (for lineage tracking)
    wandb_project : str
        W&B project for model registry (default: "viscy-model-registry")
    shared_dir : str
        Shared checkpoint directory path (default: "/hpc/models/shared")
    description : str | None
        Model description (e.g., "Trained on RPE1 collection v0.0.1, val_loss=0.15")
    metadata : dict[str, Any] | None
        Additional metadata to store (metrics, config, etc.)
    airtable_base_id : str | None
        Airtable base ID to log model to Models table (optional)
    airtable_collection_id : str | None
        Airtable collection record ID to link this model to (optional)

    Returns
    -------
    str
        W&B artifact URL

    Examples
    --------
    Register a model after training:

    >>> from viscy.airtable.register_model import register_model
    >>> artifact_url = register_model(
    ...     checkpoint_path="logs/wandb/run-20260107/checkpoints/epoch=50.ckpt",
    ...     config_path="examples/configs/fit_example.yml",  # Training config
    ...     model_name="contrastive-rpe1",
    ...     model_type="contrastive",
    ...     version="v2",
    ...     aliases=["production", "best"],
    ...     wandb_run_id="20260107-152420",
    ...     description="Trained on RPE1 collection v0.0.1, best val_loss=0.15",
    ...     metadata={"val_loss": 0.15, "collection_name": "RPE1_infection", "collection_version": "v0.0.1"},
    ...     airtable_base_id="app8vqaoWyOwa0sB5",  # Optional: log to Airtable
    ...     airtable_collection_id="recXXXXXXXXXXXXXX",  # Optional: link to collection
    ... )
    >>> print(f"Model registered: {artifact_url}")

    CLI usage:

    >>> # From command line
    >>> python -m viscy.airtable.register_model \\
    ...     logs/wandb/run-20260107/checkpoints/epoch=50.ckpt \\
    ...     --name contrastive-rpe1 \\
    ...     --type contrastive \\
    ...     --version v2 \\
    ...     --aliases production best \\
    ...     --run-id 20260107-152420 \\
    ...     --description "Best RPE1 model"
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # 1. Copy to shared directory
    shared_path = Path(shared_dir) / model_type / f"{model_name}-{version}.ckpt"
    shared_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_path, shared_path)
    print(f"✓ Copied to shared directory: {shared_path}")

    # 2. Initialize W&B (use API for non-training context)
    run = wandb.init(project=wandb_project, job_type="register-model")

    # 3. Create artifact with metadata (references only, not uploads)
    artifact_metadata = {
        "model_type": model_type,
        "version": version,
        "checkpoint_path": str(shared_path),
        "original_checkpoint": str(checkpoint_path),
    }

    # Add custom metadata
    if metadata:
        artifact_metadata.update(metadata)

    artifact = wandb.Artifact(
        name=model_name,
        type="model",
        description=description or f"{model_type} model {version}",
        metadata=artifact_metadata,
    )

    # 4. Add training config to artifact (if provided)
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            artifact.add_file(str(config_path), name="config_fit.yml")
            print(f"✓ Added training config to artifact: {config_path.name}")
        else:
            print(f"⚠ Config file not found: {config_path}")

    # 5. Store checkpoint path in metadata only (no upload/reference)
    # The checkpoint stays on HPC, W&B only tracks the metadata
    # This is "Option 2" from the plan - references only, not uploads

    # 6. Link to training run (lineage)
    if wandb_run_id:
        artifact.metadata["training_run_id"] = wandb_run_id
        # Create lineage link
        try:
            training_run = wandb.Api().run(
                f"{run.entity}/{wandb_project}/{wandb_run_id}"
            )
            artifact.metadata["training_run_url"] = training_run.url
        except Exception as e:
            print(f"⚠ Could not link to training run: {e}")

    # 7. Log artifact with version and aliases
    aliases = aliases or []
    wandb.log_artifact(artifact, aliases=aliases)

    artifact_url = (
        f"https://wandb.ai/{run.entity}/{wandb_project}/artifacts/model/{model_name}"
    )

    wandb.finish()

    print(f"✓ Registered in W&B: {model_name}:{version}")
    print(f"  Aliases: {aliases}")
    print(f"  View: {artifact_url}")

    # 8. Optionally log to Airtable Models table
    if airtable_base_id and airtable_collection_id:
        try:
            import getpass

            from viscy.airtable.database import AirtableManager

            airtable_db = AirtableManager(base_id=airtable_base_id)

            # Prepare complete metadata for Airtable (mirrors W&B artifact metadata)
            airtable_metadata = {
                # Core W&B fields
                "model_type": model_type,
                "version": version,
                "wandb_url": artifact_url,
                "original_checkpoint": str(checkpoint_path),
            }

            # Add all custom metadata (metrics, architecture, collection lineage, etc.)
            if metadata:
                airtable_metadata.update(metadata)

            model_id = airtable_db.log_model_training(
                collection_id=airtable_collection_id,
                wandb_run_id=wandb_run_id or "unknown",
                model_name=f"{model_name}:{version}",
                checkpoint_path=str(shared_path),
                trained_by=getpass.getuser(),
                metrics=airtable_metadata,  # Pass ALL metadata to Airtable
            )

            print(f"✓ Logged to Airtable Models table (record ID: {model_id})")
            print(f"  Linked to collection: {airtable_collection_id}")
            print(f"  Metadata fields passed: {len(airtable_metadata)}")

        except Exception as e:
            print(f"⚠ Could not log to Airtable: {e}")
            print("  (Fields without matching Airtable columns are ignored)")

    return artifact_url


def load_model_from_registry(
    model_name: str,
    version: str = "latest",
    wandb_project: str = "viscy-model-registry",
    model_class=None,
    **model_kwargs,
):
    """
    Load a model from W&B artifact registry.

    This fetches the checkpoint path from W&B metadata and loads the model
    from the shared HPC directory (does not download from W&B cloud).

    Parameters
    ----------
    model_name : str
        Model artifact name
    version : str
        Version or alias ("latest", "production", "v2")
    wandb_project : str
        W&B project name (default: "viscy-model-registry")
    model_class : LightningModule class
        Model class to instantiate (e.g., ContrastiveModule)
    **model_kwargs
        Additional arguments for model initialization

    Returns
    -------
    model : LightningModule
        Loaded model in eval mode

    Examples
    --------
    Load a registered model:

    >>> from viscy.representation.engine import ContrastiveModule
    >>> from viscy.airtable.register_model import load_model_from_registry
    >>>
    >>> model = load_model_from_registry(
    ...     model_name="contrastive-rpe1",
    ...     version="production",
    ...     wandb_project="viscy-model-registry",
    ...     model_class=ContrastiveModule,
    ... )
    >>> model.eval()
    >>> embeddings = model(images)
    """
    # 1. Get artifact metadata from W&B
    api = wandb.Api()
    entity = api.default_entity
    artifact = api.artifact(f"{entity}/{wandb_project}/{model_name}:{version}")

    # 2. Get checkpoint path from metadata (no download needed)
    checkpoint_path = artifact.metadata.get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError(
            f"Artifact {model_name}:{version} missing checkpoint_path metadata"
        )

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"The model is registered but the file is missing from the shared directory."
        )

    print(f"✓ Loading model from: {checkpoint_path}")
    print(f"  Artifact: {model_name}:{version}")
    print(f"  Description: {artifact.description}")

    # 3. Load model
    if model_class is None:
        raise ValueError("Must provide model_class (e.g., ContrastiveModule)")

    model = model_class.load_from_checkpoint(checkpoint_path, **model_kwargs)
    model.eval()

    return model


def list_registered_models(
    wandb_project: str = "viscy-model-registry",
    model_type: str | None = None,
) -> list[dict[str, Any]]:
    """
    List all registered models in W&B artifact registry.

    Parameters
    ----------
    wandb_project : str
        W&B project name (default: "viscy-model-registry")
    model_type : str | None
        Filter by model type (contrastive, segmentation, vae, translation)

    Returns
    -------
    list[dict[str, Any]]
        List of model metadata dictionaries

    Examples
    --------
    >>> from viscy.airtable.register_model import list_registered_models
    >>>
    >>> # List all models
    >>> models = list_registered_models()
    >>> for m in models:
    ...     print(f"{m['name']}:{m['version']} - {m['description']}")
    >>>
    >>> # List only contrastive models
    >>> contrastive = list_registered_models(model_type="contrastive")
    """
    api = wandb.Api()

    # Get all artifact collections in the project
    try:
        # Get entity from API
        entity = api.default_entity
        artifact_type = "model"
        collection_name = f"{entity}/{wandb_project}/{artifact_type}"

        # Use the artifacts API (simpler)
        artifacts = api.artifacts(artifact_type, collection_name)

    except Exception as e:
        # Project might not exist yet or no artifacts
        print(f"Warning: Could not list artifacts - {e}")
        return []

    models = []
    try:
        for artifact in artifacts:
            metadata = artifact.metadata

            # Filter by model type if specified
            if model_type and metadata.get("model_type") != model_type:
                continue

            models.append(
                {
                    "name": artifact.name,
                    "version": artifact.version,
                    "aliases": artifact.aliases,
                    "description": artifact.description,
                    "model_type": metadata.get("model_type"),
                    "checkpoint_path": metadata.get("checkpoint_path"),
                    "created_at": artifact.created_at,
                    "metadata": metadata,
                }
            )
    except Exception as e:
        print(f"Warning: Error iterating artifacts - {e}")

    return models


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Register a trained model to W&B artifact registry"
    )

    # Required arguments
    parser.add_argument("checkpoint_path", help="Path to checkpoint file (.ckpt)")
    parser.add_argument(
        "--name", required=True, help="Model name (e.g., contrastive-rpe1)"
    )
    parser.add_argument(
        "--type",
        required=True,
        choices=["contrastive", "segmentation", "vae", "translation"],
        help="Model type",
    )
    parser.add_argument("--version", required=True, help="Version (e.g., v1, v2)")

    # Optional arguments
    parser.add_argument(
        "--config",
        help="Path to training config YAML file (config_fit.yml)",
    )
    parser.add_argument(
        "--aliases",
        nargs="+",
        help="Aliases (e.g., production best latest)",
    )
    parser.add_argument(
        "--run-id",
        help="W&B run ID that trained this model",
    )
    parser.add_argument(
        "--project",
        default="viscy-model-registry",
        help="W&B project name (default: viscy-model-registry)",
    )
    parser.add_argument(
        "--shared-dir",
        default="/hpc/models/shared",
        help="Shared checkpoint directory (default: /hpc/models/shared)",
    )
    parser.add_argument(
        "--description",
        help="Model description",
    )
    parser.add_argument(
        "--airtable-base-id",
        help="Airtable base ID (optional, for logging to Models table)",
    )
    parser.add_argument(
        "--airtable-collection-id",
        help="Airtable collection record ID (optional, for linking)",
    )

    args = parser.parse_args()

    register_model(
        checkpoint_path=args.checkpoint_path,
        config_path=args.config,
        model_name=args.name,
        model_type=args.type,
        version=args.version,
        aliases=args.aliases,
        wandb_run_id=args.run_id,
        wandb_project=args.project,
        shared_dir=args.shared_dir,
        description=args.description,
        airtable_base_id=args.airtable_base_id,
        airtable_collection_id=args.airtable_collection_id,
    )
