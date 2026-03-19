"""Thin interface to the Airtable Datasets table."""

from __future__ import annotations

import os

import pandas as pd
from pyairtable import Api

from airtable_utils.schemas import DatasetRecord

TABLE_NAME = "Datasets"


class AirtableDatasets:
    """Interface to the Datasets table in the Computational Imaging Database.

    Credentials are read exclusively from environment variables:

    - ``AIRTABLE_API_KEY``: Airtable personal access token.
    - ``AIRTABLE_BASE_ID``: Airtable base ID.

    Raises
    ------
    ValueError
        If either environment variable is not set or empty.
    """

    def __init__(self) -> None:
        api_key = os.environ.get("AIRTABLE_API_KEY", "")
        base_id = os.environ.get("AIRTABLE_BASE_ID", "")
        if not api_key:
            raise ValueError(
                "AIRTABLE_API_KEY environment variable is required but not set."
            )
        if not base_id:
            raise ValueError(
                "AIRTABLE_BASE_ID environment variable is required but not set."
            )
        api = Api(api_key)
        self._table = api.table(base_id, TABLE_NAME)

    def list_records(self, filter_formula: str | None = None) -> pd.DataFrame:
        """Return all FOV records as a DataFrame.

        Parameters
        ----------
        filter_formula : str or None
            Airtable formula to filter records.
        """
        kwargs = {}
        if filter_formula:
            kwargs["formula"] = filter_formula
        raw = self._table.all(**kwargs)
        records = [DatasetRecord.from_airtable_record(r) for r in raw]
        return pd.DataFrame([r.model_dump() for r in records])

    def get_dataset_records(self, dataset_name: str) -> list[DatasetRecord]:
        """Return FOV records for a specific dataset.

        Parameters
        ----------
        dataset_name : str
            Value of the ``dataset`` field to filter on.
        """
        formula = f"{{dataset}} = '{dataset_name}'"
        raw = self._table.all(formula=formula)
        return [DatasetRecord.from_airtable_record(r) for r in raw]

    def get_unique_datasets(self) -> list[str]:
        """Return sorted unique dataset names."""
        raw = self._table.all(fields=["dataset"])
        names = {r["fields"]["dataset"] for r in raw if r.get("fields", {}).get("dataset")}
        return sorted(names)

    def batch_update(self, updates: list[dict]) -> None:
        """Batch-update records.

        Parameters
        ----------
        updates : list[dict]
            Each dict has ``"id"`` (record ID) and ``"fields"`` keys.
        """
        self._table.batch_update(updates)

    def batch_create(self, records: list[dict]) -> list[dict]:
        """Batch-create new records.

        Parameters
        ----------
        records : list[dict]
            Each dict has a ``"fields"`` key with field name/value pairs.

        Returns
        -------
        list[dict]
            Created records as returned by the Airtable API.
        """
        return self._table.batch_create([r["fields"] for r in records])
