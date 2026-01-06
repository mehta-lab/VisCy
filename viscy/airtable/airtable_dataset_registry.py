"""Dataset registry integration with Airtable for experiment tracking."""

import os
from datetime import datetime
from typing import Any

from pyairtable import Api


class AirtableDatasetRegistry:
    """
    Interface to Airtable for dataset registry.

    Airtable acts as source of truth for:
    - Dataset paths on HPC
    - Dataset versions and metadata
    - Links between datasets and trained models

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
    >>> dataset = registry.get_dataset("rpe1_fucci_embeddings", version="v2")
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
        self.datasets_table = self.api.table(base_id, "Datasets")
        self.models_table = self.api.table(base_id, "Models")

    def get_dataset(self, name: str, version: str | None = None) -> dict[str, Any]:
        """
        Retrieve dataset record from Airtable.

        Parameters
        ----------
        name : str
            Dataset name
        version : str | None
            Specific version (e.g., "v2"). If None, returns latest.

        Returns
        -------
        dict
            Airtable record with fields:
            - id: Airtable record ID
            - hpc_path: Path to dataset on HPC
            - version: Dataset version
            - sha256: Dataset hash
            - created_date: Creation timestamp
        """
        if version:
            formula = f"AND({{name}}='{name}', {{version}}='{version}')"
        else:
            formula = f"{{name}}='{name}'"

        records = self.datasets_table.all(formula=formula, sort=["-created_date"])

        if not records:
            raise ValueError(
                f"Dataset '{name}' (version={version}) not found in Airtable"
            )

        record = records[0]
        return {"id": record["id"], **record["fields"]}

    def log_model_training(
        self,
        dataset_id: str,
        mlflow_run_id: str,
        model_name: str | None = None,
        metrics: dict[str, float] | None = None,
        checkpoint_path: str | None = None,
        trained_by: str | None = None,
    ) -> str:
        """
        Log that a model was trained using a dataset.

        Creates entry in Models table and updates Datasets table.

        Parameters
        ----------
        dataset_id : str
            Airtable record ID of dataset used
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
            "dataset": [dataset_id],  # Link to dataset
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

        # Update dataset record to track usage
        dataset = self.datasets_table.get(dataset_id)
        models_trained_str = dataset["fields"].get("models_trained", "")

        # Handle models_trained as comma-separated string
        if models_trained_str:
            models_list = [m.strip() for m in models_trained_str.split(",")]
            models_list.append(mlflow_run_id)
            new_models_str = ", ".join(models_list)
        else:
            new_models_str = mlflow_run_id

        self.datasets_table.update(
            dataset_id,
            {"models_trained": new_models_str, "last_used": datetime.now().isoformat()},
        )

        return created["id"]

    def list_datasets(self, formula: str | None = None) -> list[dict]:
        """
        List all datasets in registry.

        Parameters
        ----------
        formula : str | None
            Optional Airtable formula for filtering

        Returns
        -------
        list[dict]
            List of dataset records
        """
        records = self.datasets_table.all(formula=formula, sort=["-created_date"])
        return [{"id": r["id"], **r["fields"]} for r in records]

    def get_models_for_dataset(self, dataset_id: str) -> list[dict]:
        """
        Get all models trained on a specific dataset.

        Parameters
        ----------
        dataset_id : str
            Airtable record ID of dataset

        Returns
        -------
        list[dict]
            List of model records
        """
        formula = f"FIND('{dataset_id}', ARRAYJOIN({{dataset}}))"
        records = self.models_table.all(formula=formula, sort=["-trained_date"])
        return [{"id": r["id"], **r["fields"]} for r in records]
