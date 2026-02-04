"""Airtable integration for dataset management and tracking."""

from viscy.airtable.callbacks import CollectionWandbCallback
from viscy.airtable.database import AirtableManager, CollectionDataset, Collections
from viscy.airtable.factory import (
    CollectionTripletDataModule,
    create_triplet_datamodule_from_collection,
)
from viscy.airtable.register_model import (
    list_registered_models,
    load_model_from_registry,
    register_model,
)
from viscy.airtable.schemas import DatasetRecord, ModelRecord

__all__ = [
    "AirtableManager",
    "DatasetRecord",
    "ModelRecord",
    "Collections",
    "CollectionDataset",
    "CollectionTripletDataModule",
    "CollectionWandbCallback",
    "create_triplet_datamodule_from_collection",
    "register_model",
    "load_model_from_registry",
    "list_registered_models",
]
