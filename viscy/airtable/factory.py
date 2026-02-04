"""Factory functions for creating data modules from Airtable collections."""

import os
from typing import Literal, Sequence

from lightning.pytorch import LightningDataModule
from monai.transforms import MapTransform

from viscy.airtable.database import AirtableManager, CollectionDataset, Collections
from viscy.data.combined import BatchedConcatDataModule, CachedConcatDataModule
from viscy.data.triplet import TripletDataModule


def _extract_wells_from_fov_names(fov_names: list[str]) -> list[str]:
    """
    Extract unique well IDs from FOV names.

    Parameters
    ----------
    fov_names : list[str]
        List of FOV names in format "Row/Column/FOV_idx" (e.g., "B/3/0")

    Returns
    -------
    list[str]
        Unique well IDs in format "Row/Column" (e.g., ["B/3", "C/4"])

    Examples
    --------
    >>> _extract_wells_from_fov_names(["B/3/0", "B/3/1", "C/4/2"])
    ['B/3', 'C/4']
    """
    wells = set()
    for fov_name in fov_names:
        # Split "B/3/0" -> ["B", "3", "0"]
        parts = fov_name.split("/")
        if len(parts) >= 2:
            # Extract "B/3"
            well_id = f"{parts[0]}/{parts[1]}"
            wells.add(well_id)
        else:
            raise ValueError(
                f"Invalid FOV name format: '{fov_name}'. "
                f"Expected 'Row/Column/FOV_idx' (e.g., 'B/3/0')"
            )

    return sorted(list(wells))


def create_triplet_datamodule_from_collection(
    collection: Collections | CollectionDataset,
    source_channel: str | Sequence[str],
    z_range: tuple[int, int],
    *,
    initial_yx_patch_size: tuple[int, int] = (512, 512),
    final_yx_patch_size: tuple[int, int] = (224, 224),
    split_ratio: float = 0.8,
    batch_size: int = 16,
    num_workers: int = 1,
    normalizations: list[MapTransform] | None = None,
    augmentations: list[MapTransform] | None = None,
    augment_validation: bool = True,
    caching: bool = False,
    fit_include_wells: list[str] | None = None,
    fit_exclude_fovs: list[str] | None = None,
    predict_cells: bool = False,
    include_fov_names: list[str] | None = None,
    include_track_ids: list[int] | None = None,
    time_interval: Literal["any"] | int = "any",
    return_negative: bool = True,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
    pin_memory: bool = False,
    z_window_size: int | None = None,
    cache_pool_bytes: int = 0,
    use_cached_concat: bool = False,
) -> LightningDataModule:
    """
    Create TripletDataModule(s) from Airtable collection.

    Automatically handles single or multiple HCS plates:
    - Single plate: Returns TripletDataModule
    - Multiple plates: Returns BatchedConcatDataModule or CachedConcatDataModule

    Parameters
    ----------
    collection : Collections | CollectionDataset
        Collections from AirtableManager.get_dataset_paths() or single CollectionDataset
    source_channel : str | Sequence[str]
        Input channel name(s) - REQUIRED
    z_range : tuple[int, int]
        Range of valid z-slices - REQUIRED
    initial_yx_patch_size : tuple[int, int]
        YX size of initially sampled patch, default (512, 512)
    final_yx_patch_size : tuple[int, int]
        Output patch size after augmentation, default (224, 224)
    split_ratio : float
        Train/val split ratio, default 0.8
    batch_size : int
        Batch size, default 16
    num_workers : int
        Number of dataloader workers, default 1
    normalizations : list[MapTransform] | None
        Normalization transforms
    augmentations : list[MapTransform] | None
        Augmentation transforms
    augment_validation : bool
        Apply augmentations to validation, default True
    caching : bool
        Cache dataset, default False
    fit_include_wells : list[str] | None
        Override collection FOVs with specific wells (e.g., ["B/3", "C/4"]).
        Takes precedence over collection.fov_names
    fit_exclude_fovs : list[str] | None
        Exclude specific FOV paths from collection
    predict_cells : bool
        Predict on specific cells only, default False
    include_fov_names : list[str] | None
        FOV names for prediction (when predict_cells=True)
    include_track_ids : list[int] | None
        Track IDs for prediction (when predict_cells=True)
    time_interval : Literal["any"] | int
        Time interval for positive sampling, default "any"
    return_negative : bool
        Return negative samples for triplet loss, default True
    persistent_workers : bool
        Keep workers alive between epochs, default False
    prefetch_factor : int | None
        Batches to prefetch per worker
    pin_memory : bool
        Pin memory for faster GPU transfer, default False
    z_window_size : int | None
        Final Z window size (inferred from z_range if None)
    cache_pool_bytes : int
        Tensorstore cache pool size in bytes, default 0
    use_cached_concat : bool
        Use CachedConcatDataModule instead of BatchedConcatDataModule
        for multi-plate collections, default False

    Returns
    -------
    LightningDataModule
        - TripletDataModule for single plate
        - BatchedConcatDataModule for multiple plates (default)
        - CachedConcatDataModule for multiple plates (if use_cached_concat=True)

    Raises
    ------
    ValueError
        - If collection has no datasets
        - If paths don't exist (validation fails)
        - If fit_include_wells and collection both specify FOVs (ambiguous)
    FileNotFoundError
        If data_path or tracks_path don't exist
    TypeError
        If collection is not Collections or CollectionDataset

    Examples
    --------
    Basic usage with single-plate collection:

    >>> from viscy.airtable.database import AirtableManager
    >>> from viscy.airtable.factory import create_triplet_datamodule_from_collection
    >>>
    >>> airtable_db = AirtableManager(base_id="appXXXXXXXXXXXXXX")
    >>> collection = airtable_db.get_dataset_paths("my_collection", "0.0.1")
    >>>
    >>> dm = create_triplet_datamodule_from_collection(
    ...     collection=collection,
    ...     source_channel=["Phase3D"],
    ...     z_range=(0, 5),
    ...     batch_size=32,
    ...     num_workers=8,
    ... )
    >>>
    >>> # Use with PyTorch Lightning
    >>> trainer.fit(model, dm)

    Multi-plate collection with normalization:

    >>> from viscy.transforms import NormalizeSampled
    >>>
    >>> collection = airtable_db.get_dataset_paths("multi_plate_collection", "1.0.0")
    >>> print(f"Collections has {len(collection)} plates")  # e.g., 3 plates
    >>>
    >>> dm = create_triplet_datamodule_from_collection(
    ...     collection=collection,
    ...     source_channel=["Phase3D", "RFP"],
    ...     z_range=(0, 10),
    ...     normalizations=[
    ...         NormalizeSampled(
    ...             ["Phase3D"],
    ...             level="fov_statistics",
    ...             subtrahend="mean",
    ...             divisor="std"
    ...         )
    ...     ],
    ...     batch_size=16,
    ...     use_cached_concat=False,  # Use BatchedConcatDataModule
    ... )
    >>> # Returns BatchedConcatDataModule wrapping 3 TripletDataModules

    Override collection FOVs with specific wells:

    >>> dm = create_triplet_datamodule_from_collection(
    ...     collection=collection,
    ...     source_channel=["Phase3D"],
    ...     z_range=(0, 5),
    ...     fit_include_wells=["B/3", "B/4"],  # Override collection FOVs
    ...     batch_size=16,
    ... )

    Using a single CollectionDataset directly:

    >>> ds = collection.datasets[0]  # Single plate
    >>> dm = create_triplet_datamodule_from_collection(
    ...     collection=ds,  # Pass CollectionDataset directly
    ...     source_channel=["Phase3D"],
    ...     z_range=(0, 5),
    ...     batch_size=16,
    ... )

    Notes
    -----
    - FOV filtering priority: fit_include_wells > collection.fov_names
    - If both specified, raises ValueError to avoid ambiguity
    - The factory validates paths before creating data modules
    - Multi-plate handling uses BatchedConcatDataModule for efficient batching
    - All TripletDataModule parameters can be passed through kwargs
    """
    # STEP 1: Normalize input - handle both Collections and CollectionDataset
    if isinstance(collection, CollectionDataset):
        datasets = [collection]
        collection_name = "single_dataset"
    elif isinstance(collection, Collections):
        if len(collection.datasets) == 0:
            raise ValueError(f"Collections '{collection.name}' has no datasets")
        datasets = collection.datasets
        collection_name = collection.name
    else:
        raise TypeError(
            f"Expected Collections or CollectionDataset, got {type(collection).__name__}"
        )

    # STEP 2: Validate all paths exist (fail early)
    for i, ds in enumerate(datasets):
        try:
            ds.validate()
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Collections '{collection_name}' dataset {i}: {e}")

    # STEP 3: Handle FOV filtering logic
    # Check for ambiguous FOV specification
    has_collection_fovs = any(len(ds.fov_names) > 0 for ds in datasets)

    if fit_include_wells is not None and has_collection_fovs:
        # Ambiguous: both collection and user specified FOVs
        raise ValueError(
            "Cannot specify both 'fit_include_wells' and use collection FOV filtering. "
            "The collection already specifies FOVs to include. "
            "Either:\n"
            "  1. Use fit_include_wells=None to respect collection FOVs, OR\n"
            "  2. Create a new collection without FOV filtering if you want custom wells"
        )

    # STEP 4: Ensure normalizations and augmentations are lists
    normalizations = normalizations or []
    augmentations = augmentations or []

    # STEP 5: Create TripletDataModule for each dataset
    data_modules = []

    for ds in datasets:
        # Determine well filtering strategy
        if fit_include_wells is not None:
            # User override: use explicit wells
            include_wells = fit_include_wells
        elif len(ds.fov_names) > 0:
            # Convert collection FOV names to well IDs
            include_wells = _extract_wells_from_fov_names(ds.fov_names)
        else:
            # No filtering: use all wells
            include_wells = None

        # Create TripletDataModule
        dm = TripletDataModule(
            data_path=ds.data_path,
            tracks_path=ds.tracks_path,
            source_channel=source_channel,
            z_range=z_range,
            initial_yx_patch_size=initial_yx_patch_size,
            final_yx_patch_size=final_yx_patch_size,
            split_ratio=split_ratio,
            batch_size=batch_size,
            num_workers=num_workers,
            normalizations=normalizations,
            augmentations=augmentations,
            augment_validation=augment_validation,
            caching=caching,
            fit_include_wells=include_wells,
            fit_exclude_fovs=fit_exclude_fovs,
            predict_cells=predict_cells,
            include_fov_names=include_fov_names,
            include_track_ids=include_track_ids,
            time_interval=time_interval,
            return_negative=return_negative,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            z_window_size=z_window_size,
            cache_pool_bytes=cache_pool_bytes,
        )
        data_modules.append(dm)

    # STEP 6: Return appropriate data module type
    if len(data_modules) == 1:
        # Single plate: return TripletDataModule directly
        return data_modules[0]
    else:
        # Multiple plates: wrap in ConcatDataModule
        if use_cached_concat:
            return CachedConcatDataModule(data_modules=data_modules)
        else:
            return BatchedConcatDataModule(data_modules=data_modules)


class CollectionTripletDataModule(TripletDataModule):
    """
    TripletDataModule that fetches paths from Airtable collections.

    This class is designed to work with PyTorch Lightning CLI and config files.
    It extends TripletDataModule to accept Airtable collection parameters instead
    of explicit data_path and tracks_path.

    Parameters
    ----------
    base_id : str
        Airtable base ID
    collection_name : str
        Name of the collection in Airtable
    collection_version : str
        Semantic version of the collection (e.g., "0.0.1")
    source_channel : str | Sequence[str]
        Input channel name(s)
    z_range : tuple[int, int]
        Range of valid z-slices
    api_key : str | None
        Airtable API key (if None, reads from AIRTABLE_API_KEY env var)
    **kwargs
        All other parameters passed to TripletDataModule.__init__

    Raises
    ------
    ValueError
        If collection has multiple datasets (only single-plate collections supported)

    Examples
    --------
    In a Lightning config file (config.yml):

    ```yaml
    data:
      class_path: viscy.airtable.factory.CollectionTripletDataModule
      init_args:
        base_id: "appXXXXXXXXXXXXXX"
        collection_name: "my_collection"
        collection_version: "0.0.1"
        source_channel: [Phase]
        z_range: [0, 5]
        batch_size: 16
        num_workers: 8
        normalizations:
          - class_path: viscy.transforms.NormalizeSampled
            init_args:
              keys: [Phase]
              level: fov_statistics
              subtrahend: mean
              divisor: std
    ```

    Command line usage:
    ```bash
    viscy fit -c config.yml
    ```

    Direct usage in Python:
    ```python
    dm = CollectionTripletDataModule(
        base_id="appXXXXXXXXXXXXXX",
        collection_name="my_collection",
        collection_version="0.0.1",
        source_channel=["Phase"],
        z_range=(0, 5),
        batch_size=16,
    )
    trainer.fit(model, dm)
    ```

    Notes
    -----
    - Only supports single-plate collections (use create_triplet_datamodule_from_collection
      for multi-plate support with BatchedConcatDataModule)
    - Fetches collection from Airtable during __init__
    - All TripletDataModule parameters are available
    - FOV filtering from collection is automatically applied via fit_include_wells
    """

    def __init__(
        self,
        base_id: str,
        collection_name: str,
        collection_version: str,
        source_channel: str | Sequence[str],
        z_range: tuple[int, int],
        api_key: str | None = None,
        fit_include_wells: list[str] | None = None,
        **kwargs,
    ):
        # Fetch collection from Airtable
        airtable_db = AirtableManager(
            base_id=base_id, api_key=api_key or os.getenv("AIRTABLE_API_KEY")
        )
        collection = airtable_db.get_dataset_paths(
            collection_name=collection_name,
            version=collection_version,
        )

        # Validate single plate
        if len(collection.datasets) != 1:
            raise ValueError(
                f"CollectionTripletDataModule only supports single-plate collections. "
                f"Collections '{collection_name}' has {len(collection.datasets)} plates. "
                f"Use create_triplet_datamodule_from_collection() for multi-plate support."
            )

        dataset = collection.datasets[0]

        # Store collection metadata as instance attributes for callbacks/logging
        self.base_id = base_id
        self.collection_name = collection_name
        self.collection_version = collection_version
        self.data_path = dataset.data_path
        self.tracks_path = dataset.tracks_path

        # Handle FOV filtering
        if fit_include_wells is not None:
            # User override: use explicit wells
            include_wells = fit_include_wells
        elif len(dataset.fov_names) > 0:
            # Convert collection FOV names to well IDs
            include_wells = _extract_wells_from_fov_names(dataset.fov_names)
        else:
            # No filtering: use all wells
            include_wells = None

        # Initialize parent TripletDataModule with extracted paths
        super().__init__(
            data_path=dataset.data_path,
            tracks_path=dataset.tracks_path,
            source_channel=source_channel,
            z_range=z_range,
            fit_include_wells=include_wells,
            **kwargs,
        )
