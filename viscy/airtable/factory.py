"""Factory functions for creating data modules from Airtable manifests."""

import os
from typing import Literal, Sequence

from lightning.pytorch import LightningDataModule
from monai.transforms import MapTransform

from viscy.airtable.database import AirtableManager, Manifest, ManifestDataset
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


def create_triplet_datamodule_from_manifest(
    manifest: Manifest | ManifestDataset,
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
    Create TripletDataModule(s) from Airtable manifest.

    Automatically handles single or multiple HCS plates:
    - Single plate: Returns TripletDataModule
    - Multiple plates: Returns BatchedConcatDataModule or CachedConcatDataModule

    Parameters
    ----------
    manifest : Manifest | ManifestDataset
        Manifest from AirtableManager.get_dataset_paths() or single ManifestDataset
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
        Override manifest FOVs with specific wells (e.g., ["B/3", "C/4"]).
        Takes precedence over manifest.fov_names
    fit_exclude_fovs : list[str] | None
        Exclude specific FOV paths from manifest
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
        for multi-plate manifests, default False

    Returns
    -------
    LightningDataModule
        - TripletDataModule for single plate
        - BatchedConcatDataModule for multiple plates (default)
        - CachedConcatDataModule for multiple plates (if use_cached_concat=True)

    Raises
    ------
    ValueError
        - If manifest has no datasets
        - If paths don't exist (validation fails)
        - If fit_include_wells and manifest both specify FOVs (ambiguous)
    FileNotFoundError
        If data_path or tracks_path don't exist
    TypeError
        If manifest is not Manifest or ManifestDataset

    Examples
    --------
    Basic usage with single-plate manifest:

    >>> from viscy.airtable.database import AirtableManager
    >>> from viscy.airtable.factory import create_triplet_datamodule_from_manifest
    >>>
    >>> airtable_db = AirtableManager(base_id="appXXXXXXXXXXXXXX")
    >>> manifest = airtable_db.get_dataset_paths("my_manifest", "0.0.1")
    >>>
    >>> dm = create_triplet_datamodule_from_manifest(
    ...     manifest=manifest,
    ...     source_channel=["Phase3D"],
    ...     z_range=(0, 5),
    ...     batch_size=32,
    ...     num_workers=8,
    ... )
    >>>
    >>> # Use with PyTorch Lightning
    >>> trainer.fit(model, dm)

    Multi-plate manifest with normalization:

    >>> from viscy.transforms import NormalizeSampled
    >>>
    >>> manifest = airtable_db.get_dataset_paths("multi_plate_manifest", "1.0.0")
    >>> print(f"Manifest has {len(manifest)} plates")  # e.g., 3 plates
    >>>
    >>> dm = create_triplet_datamodule_from_manifest(
    ...     manifest=manifest,
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

    Override manifest FOVs with specific wells:

    >>> dm = create_triplet_datamodule_from_manifest(
    ...     manifest=manifest,
    ...     source_channel=["Phase3D"],
    ...     z_range=(0, 5),
    ...     fit_include_wells=["B/3", "B/4"],  # Override manifest FOVs
    ...     batch_size=16,
    ... )

    Using a single ManifestDataset directly:

    >>> ds = manifest.datasets[0]  # Single plate
    >>> dm = create_triplet_datamodule_from_manifest(
    ...     manifest=ds,  # Pass ManifestDataset directly
    ...     source_channel=["Phase3D"],
    ...     z_range=(0, 5),
    ...     batch_size=16,
    ... )

    Notes
    -----
    - FOV filtering priority: fit_include_wells > manifest.fov_names
    - If both specified, raises ValueError to avoid ambiguity
    - The factory validates paths before creating data modules
    - Multi-plate handling uses BatchedConcatDataModule for efficient batching
    - All TripletDataModule parameters can be passed through kwargs
    """
    # STEP 1: Normalize input - handle both Manifest and ManifestDataset
    if isinstance(manifest, ManifestDataset):
        datasets = [manifest]
        manifest_name = "single_dataset"
    elif isinstance(manifest, Manifest):
        if len(manifest.datasets) == 0:
            raise ValueError(f"Manifest '{manifest.name}' has no datasets")
        datasets = manifest.datasets
        manifest_name = manifest.name
    else:
        raise TypeError(
            f"Expected Manifest or ManifestDataset, got {type(manifest).__name__}"
        )

    # STEP 2: Validate all paths exist (fail early)
    for i, ds in enumerate(datasets):
        try:
            ds.validate()
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Manifest '{manifest_name}' dataset {i}: {e}")

    # STEP 3: Handle FOV filtering logic
    # Check for ambiguous FOV specification
    has_manifest_fovs = any(len(ds.fov_names) > 0 for ds in datasets)

    if fit_include_wells is not None and has_manifest_fovs:
        # Ambiguous: both manifest and user specified FOVs
        raise ValueError(
            "Cannot specify both 'fit_include_wells' and use manifest FOV filtering. "
            "The manifest already specifies FOVs to include. "
            "Either:\n"
            "  1. Use fit_include_wells=None to respect manifest FOVs, OR\n"
            "  2. Create a new manifest without FOV filtering if you want custom wells"
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
            # Convert manifest FOV names to well IDs
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


class ManifestTripletDataModule(TripletDataModule):
    """
    TripletDataModule that fetches paths from Airtable manifests.

    This class is designed to work with PyTorch Lightning CLI and config files.
    It extends TripletDataModule to accept Airtable manifest parameters instead
    of explicit data_path and tracks_path.

    Parameters
    ----------
    base_id : str
        Airtable base ID
    manifest_name : str
        Name of the manifest in Airtable
    manifest_version : str
        Semantic version of the manifest (e.g., "0.0.1")
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
        If manifest has multiple datasets (only single-plate manifests supported)

    Examples
    --------
    In a Lightning config file (config.yml):

    ```yaml
    data:
      class_path: viscy.airtable.factory.ManifestTripletDataModule
      init_args:
        base_id: "appXXXXXXXXXXXXXX"
        manifest_name: "my_manifest"
        manifest_version: "0.0.1"
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
    dm = ManifestTripletDataModule(
        base_id="appXXXXXXXXXXXXXX",
        manifest_name="my_manifest",
        manifest_version="0.0.1",
        source_channel=["Phase"],
        z_range=(0, 5),
        batch_size=16,
    )
    trainer.fit(model, dm)
    ```

    Notes
    -----
    - Only supports single-plate manifests (use create_triplet_datamodule_from_manifest
      for multi-plate support with BatchedConcatDataModule)
    - Fetches manifest from Airtable during __init__
    - All TripletDataModule parameters are available
    - FOV filtering from manifest is automatically applied via fit_include_wells
    """

    def __init__(
        self,
        base_id: str,
        manifest_name: str,
        manifest_version: str,
        source_channel: str | Sequence[str],
        z_range: tuple[int, int],
        api_key: str | None = None,
        fit_include_wells: list[str] | None = None,
        **kwargs,
    ):
        # Fetch manifest from Airtable
        airtable_db = AirtableManager(
            base_id=base_id, api_key=api_key or os.getenv("AIRTABLE_API_KEY")
        )
        manifest = airtable_db.get_dataset_paths(
            manifest_name=manifest_name,
            version=manifest_version,
        )

        # Validate single plate
        if len(manifest.datasets) != 1:
            raise ValueError(
                f"ManifestTripletDataModule only supports single-plate manifests. "
                f"Manifest '{manifest_name}' has {len(manifest.datasets)} plates. "
                f"Use create_triplet_datamodule_from_manifest() for multi-plate support."
            )

        dataset = manifest.datasets[0]

        # Store manifest metadata as instance attributes for callbacks/logging
        self.base_id = base_id
        self.manifest_name = manifest_name
        self.manifest_version = manifest_version
        self.data_path = dataset.data_path
        self.tracks_path = dataset.tracks_path

        # Handle FOV filtering
        if fit_include_wells is not None:
            # User override: use explicit wells
            include_wells = fit_include_wells
        elif len(dataset.fov_names) > 0:
            # Convert manifest FOV names to well IDs
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
