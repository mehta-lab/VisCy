"""Airtable integration for dataset management and tracking."""

from viscy.airtable.callbacks import ManifestWandbCallback
from viscy.airtable.database import AirtableManager, Manifest, ManifestDataset
from viscy.airtable.factory import (
    ManifestTripletDataModule,
    create_triplet_datamodule_from_manifest,
)
from viscy.airtable.register_model import (
    list_registered_models,
    load_model_from_registry,
    register_model,
)
from viscy.airtable.schemas import DatasetRecord

__all__ = [
    "AirtableManager",
    "DatasetRecord",
    "Manifest",
    "ManifestDataset",
    "ManifestTripletDataModule",
    "ManifestWandbCallback",
    "create_triplet_datamodule_from_manifest",
    "register_model",
    "load_model_from_registry",
    "list_registered_models",
]
