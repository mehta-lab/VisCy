"""Dataset registry integration with Airtable for experiment tracking."""

import os
from datetime import datetime
from typing import Any

from pyairtable import Api


class AirtableManifests:
    """
    Interface to Airtable for manifests.

    Airtable acts as source of truth for:
    - Dataset manifests

    Parameters
    ----------
    base_id : str
        Airtable base ID
    api_key : str | None
        Airtable API key. If None, reads from AIRTABLE_API_KEY env var.

    Examples
    --------
    >>> registry = AirtableDatasetRegistry(base_id="appXXXXXXXXXXXXXX")
    >>>
    >>> # Get dataset info
    >>> dataset = registry.get_manifest("rpe1_fucci_embeddings", version="v2")
    >>> print(dataset['hpc_path'])
    >>>
    >>> # Record that a model was trained with this dataset
    >>> registry.log_model_training(
    ...     dataset_id=dataset['id'],
    ...     mlflow_run_id="run_123",
    ...     metrics={"accuracy": 0.89}
    ... )
    """

    def __init__(
        self,
        base_id: str,
        api_key: str | None = None,
    ):
        api_key = api_key or os.getenv("AIRTABLE_API_KEY")
        if not api_key:
            raise ValueError("Airtable API key required (set AIRTABLE_API_KEY)")

        self.api = Api(api_key)
        self.base_id = base_id
        self.manifests_table = self.api.table(base_id, "Manifest")
        self.models_table = self.api.table(base_id, "Models")

    def get_manifest(self, name: str, version: str | None = None) -> dict[str, Any]:
        """
        Retrieve manifest record from Airtable.

        Parameters
        ----------
        name : str
            Manifest name
        version : str | None
            Specific version (e.g., "0.0.1"). If None, returns latest.

        Returns
        -------
        dict
            Manifest record with fields from MANIFESTS_INDEX
        """
        if version:
            formula = f"AND({{name}}='{name}', {{version}}='{version}')"
        else:
            formula = f"{{name}}='{name}'"

        records = self.manifests_table.all(formula=formula, sort=["-created_time"])

        if not records:
            raise ValueError(
                f"Manifest '{name}' (version={version}) not found in Airtable"
            )

        record = records[0]
        return {"id": record["id"], **record["fields"]}

    def log_model_training(
        self,
        manifest_id: str,
        mlflow_run_id: str,
        model_name: str | None = None,
        metrics: dict[str, float] | None = None,
        checkpoint_path: str | None = None,
        trained_by: str | None = None,
    ) -> str:
        """
        Log that a model was trained using a manifest.

        Creates entry in Models table and updates Manifests table.

        Parameters
        ----------
        manifest_id : str
            Airtable record ID of manifest used
        mlflow_run_id : str
            MLflow run ID for experiment tracking
        model_name : str | None
            Human-readable model name
        metrics : dict | None
            Training metrics (accuracy, f1_score, etc.)
        checkpoint_path : str | None
            Path to saved model checkpoint
        trained_by : str | None
            Username of person who trained the model

        Returns
        -------
        str
            Airtable record ID of created model entry
        """
        # Create model record
        model_record = {
            "model_name": model_name or f"model_{datetime.now():%Y%m%d_%H%M%S}",
            "manifest": [manifest_id],  # Link to manifest
            "mlflow_run_id": mlflow_run_id,
            "trained_date": datetime.now().isoformat(),
        }

        if metrics:
            model_record.update(metrics)

        if checkpoint_path:
            model_record["checkpoint_path"] = checkpoint_path

        if trained_by:
            model_record["trained_by"] = trained_by

        created = self.models_table.create(model_record)

        # Update manifest record to track usage
        manifest = self.manifests_table.get(manifest_id)
        models_trained_str = manifest["fields"].get("models_trained", "")

        # Handle models_trained as comma-separated string
        if models_trained_str:
            models_list = [m.strip() for m in models_trained_str.split(",")]
            models_list.append(mlflow_run_id)
            new_models_str = ", ".join(models_list)
        else:
            new_models_str = mlflow_run_id

        self.manifests_table.update(
            manifest_id,
            {"models_trained": new_models_str, "last_used": datetime.now().isoformat()},
        )

        return created["id"]

    def list_manifests(self, formula: str | None = None) -> list[dict]:
        """
        List all manifests in registry.

        Parameters
        ----------
        formula : str | None
            Optional Airtable formula for filtering

        Returns
        -------
        list[dict]
            List of manifest records
        """
        records = self.manifests_table.all(formula=formula, sort=["-created_time"])
        return [{"id": r["id"], **r["fields"]} for r in records]

    def get_models_for_manifest(self, manifest_id: str) -> list[dict]:
        """
        Get all models trained on a specific manifest.

        Parameters
        ----------
        manifest_id : str
            Airtable record ID of manifest

        Returns
        -------
        list[dict]
            List of model records
        """
        formula = f"FIND('{manifest_id}', ARRAYJOIN({{manifest}}))"
        records = self.models_table.all(formula=formula, sort=["-trained_date"])
        return [{"id": r["id"], **r["fields"]} for r in records]
