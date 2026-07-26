"""Snapshot ``(dataset, target)`` selectors into a frozen ``BenchmarkCollection`` YAML.

The freezer reads ground-truth channel names and FOV membership from
the train/test zarr stores (not from the manifest), so the resulting
collection describes what is actually on disk at freeze time.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from iohub.ngff import open_ome_zarr

from dynacell.data import (
    BenchmarkCollection,
    TargetConfig,
    VoxelSpacing,
    get_manifest,
)
from dynacell.data.collections import CollectionExperiment, Provenance
from viscy_data.collection import ChannelEntry

_STORE_ROLES: tuple[str, ...] = ("train", "test")


@dataclass(frozen=True, slots=True)
class ExperimentSelector:
    """One ``(dataset, target)`` entry in a freeze request.

    Parameters
    ----------
    dataset
        Registered dataset key (e.g. ``"aics-hipsc"``,
        ``"a549-mantis-2024_11_07"``).
    target
        Target name within the dataset manifest (e.g. ``"sec61b"``).
    include_train
        If ``False``, skip this selector's train store during freeze.
    include_test
        If ``False``, skip this selector's test store during freeze.
    condition
        If set, keep only positions whose ``.zattrs["condition"]`` equals
        this value. Used to author condition-filtered joint collections
        (e.g. ``"mock"`` for uninfected-only A549 regimes). When ``None``,
        every position in the referenced store contributes.
    """

    dataset: str
    target: str
    include_train: bool = True
    include_test: bool = True
    condition: str | None = None


def _dataset_short_name(dataset: str) -> str:
    """Derive an experiment-name-safe short tag from a dataset key.

    ``aics-hipsc`` becomes ``hipsc``; ``a549-mantis`` becomes
    ``a549_mantis``. The short tag is embedded in experiment names so
    joint collections (multiple datasets, same target) disambiguate
    cleanly.
    """
    stem = dataset.removeprefix("aics-")
    return stem.replace("-", "_")


def freeze_collection(
    experiments: list[ExperimentSelector],
    output_path: Path,
    *,
    created_by: str,
    created_at: str | None = None,
    mirror_path: Path | None = None,
    description: str | None = None,
) -> BenchmarkCollection:
    """Freeze a list of selectors into a YAML collection file.

    Parameters
    ----------
    experiments
        One or more ``ExperimentSelector`` instances. Each selector
        contributes up to two stores (train and test) to the collection,
        controlled by its ``include_train`` / ``include_test`` flags.
    output_path
        Canonical YAML location to write (typically
        ``configs/collections/virtual_staining/<name>.yaml``). The
        collection's ``name`` field is derived from ``output_path.stem``,
        so the filename must match the intended collection name.
    created_by
        Author attribution stored in ``provenance.created_by``. Required.
    created_at
        ISO-8601 timestamp for ``provenance.created_at``. Defaults to
        ``datetime.now(tz=UTC).isoformat()``. Accept an explicit value
        to reproduce an existing collection byte-for-byte.
    mirror_path
        Optional second path to receive the byte-identical YAML (used
        for the packaged ``dynacell/_configs/collections/`` mirror).
    description
        Optional collection description. When ``None``, a default is
        derived from the selectors by :func:`_resolve_description`.

    Returns
    -------
    BenchmarkCollection
        The validated collection that was written.

    Raises
    ------
    ValueError
        If ``experiments`` is empty, contains exact duplicate selectors,
        contains a no-op selector with both include flags disabled,
        would emit the same ``(dataset, target, role)`` from more than
        one selector (which would produce duplicate experiment names),
        or contributes no experiments at all because every requested
        role resolved to a missing store.
    """
    if not experiments:
        raise ValueError("experiments must be non-empty")
    if len(set(experiments)) != len(experiments):
        raise ValueError("duplicate ExperimentSelector in experiments list")
    for sel in experiments:
        if not sel.include_train and not sel.include_test:
            raise ValueError(
                f"selector {sel!r} has both include_train=False and include_test=False; drop the selector instead"
            )
    seen_contributions: set[tuple[str, str, str]] = set()
    for sel in experiments:
        for role in _STORE_ROLES:
            if role == "train" and not sel.include_train:
                continue
            if role == "test" and not sel.include_test:
                continue
            key = (sel.dataset, sel.target, role)
            if key in seen_contributions:
                raise ValueError(
                    "selectors contribute the same "
                    f"(dataset={sel.dataset!r}, target={sel.target!r}, "
                    f"role={role!r}) more than once"
                )
            seen_contributions.add(key)

    collection_experiments: list[CollectionExperiment] = []
    train_fovs: list[str] = []
    test_fovs: list[str] = []

    for sel in experiments:
        manifest = get_manifest(sel.dataset)
        short = _dataset_short_name(sel.dataset)
        target_cfg = manifest.targets[sel.target]
        for role in _STORE_ROLES:
            if role == "train" and not sel.include_train:
                continue
            if role == "test" and not sel.include_test:
                continue
            store_path = getattr(target_cfg.stores, role, None)
            if store_path is None:
                continue
            experiment_name = f"{sel.target}_{short}_{role}"
            experiment, fovs = _snapshot_store(
                store_path=Path(store_path),
                experiment_name=experiment_name,
                target_cfg=target_cfg,
                manifest_spacing=manifest.spacing,
                condition_filter=sel.condition,
            )
            collection_experiments.append(experiment)
            if role == "train":
                train_fovs.extend(fovs)
            else:
                test_fovs.extend(fovs)

    if not collection_experiments:
        pairs = ", ".join(f"({sel.dataset!r}, {sel.target!r})" for sel in experiments)
        raise ValueError(
            f"selectors [{pairs}] contributed no experiments: every requested role was either disabled by an "
            "include flag or absent from the manifest. Eval-only datasets (e.g. 'hek-mantis-*') carry "
            "stores.train=None, so a selector with include_test=False leaves nothing to snapshot. An empty "
            "collection would pass validation and then break downstream FOV lookup, which indexes experiments "
            "by name."
        )

    names = [exp.name for exp in collection_experiments]
    if len(set(names)) != len(names):
        raise ValueError(
            f"duplicate CollectionExperiment.name in {names!r}; downstream FOV validation indexes experiments by name"
        )

    provenance = Provenance(
        created_at=created_at or datetime.now(tz=UTC).isoformat(),
        created_by=created_by,
        record_ids=[],
    )
    collection = BenchmarkCollection(
        name=output_path.stem,
        description=description or _resolve_description(experiments),
        provenance=provenance,
        experiments=collection_experiments,
        train_fovs=train_fovs or None,
        test_fovs=test_fovs or None,
    )

    payload = yaml.safe_dump(
        collection.model_dump(mode="json"),
        sort_keys=False,
        default_flow_style=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload)
    if mirror_path is not None:
        mirror_path.parent.mkdir(parents=True, exist_ok=True)
        mirror_path.write_text(payload)

    return collection


def _resolve_description(experiments: list[ExperimentSelector]) -> str:
    """Pick a default description when the caller didn't supply one.

    When every selector shares one ``(dataset, target)`` pair, delegate
    to :func:`_default_description` with those scalars so single-selector
    freezes reproduce the pre-A2 template byte-for-byte. Multi-source
    single-target and multi-target collections get their own deterministic
    templates. If every condition-filtered selector agrees on one
    ``condition`` value, the template names the filter so downstream
    readers can identify mock-only / DENV-only / etc. collections at a
    glance.
    """
    pairs = {(s.dataset, s.target) for s in experiments}
    if len(pairs) == 1:
        dataset, target = next(iter(pairs))
        return _default_description(dataset, target)
    conditions = {s.condition for s in experiments if s.condition is not None}
    condition_note = ""
    if len(conditions) == 1:
        (only_condition,) = conditions
        condition_note = f" with condition filter {only_condition!r}"
    targets = sorted({s.target for s in experiments})
    if len(targets) == 1:
        datasets = sorted({s.dataset for s in experiments})
        return (
            f"Joint frozen benchmark collection for target {targets[0]!r}"
            f"{condition_note} across {len(datasets)} source datasets. "
            "Channels and FOV membership captured from the live train/test "
            "zarr stores at freeze time."
        )
    return (
        f"Joint frozen benchmark collection{condition_note} across "
        f"{len(experiments)} selectors covering targets "
        f"{', '.join(repr(t) for t in targets)}. "
        "Channels and FOV membership captured from the live train/test "
        "zarr stores at freeze time."
    )


def _snapshot_store(
    store_path: Path,
    experiment_name: str,
    target_cfg: TargetConfig,
    manifest_spacing: VoxelSpacing,
    condition_filter: str | None = None,
) -> tuple[CollectionExperiment, list[str]]:
    """Open one HCS zarr and capture channels + positions.

    Sorts positions lexicographically so fresh freezes reproduce
    committed YAMLs byte-for-byte. Asserts channel lists are
    consistent across positions. When ``condition_filter`` is set,
    keeps only positions whose ``.zattrs["condition"]`` matches; raises
    if no position carries the ``condition`` attr (wrong store) or if
    the filter removes every position (empty selector).
    """
    with open_ome_zarr(store_path, mode="r") as plate:
        positions = sorted(plate.positions(), key=lambda pair: pair[0])
        if not positions:
            raise ValueError(f"No positions in {store_path}")
        if condition_filter is not None:
            annotated = [(name, pos) for name, pos in positions if "condition" in pos.zattrs]
            if not annotated:
                raise ValueError(
                    f"Selector requested condition={condition_filter!r} but no "
                    f"position in {store_path} carries a 'condition' zattr"
                )
            positions = [(name, pos) for name, pos in annotated if pos.zattrs["condition"] == condition_filter]
            if not positions:
                raise ValueError(
                    f"Condition filter {condition_filter!r} matched 0 "
                    f"positions in {store_path}; drop the selector or "
                    "pick a condition present in this store"
                )
        channel_names = list(positions[0][1].channel_names)
        for name, pos in positions[1:]:
            if list(pos.channel_names) != channel_names:
                raise ValueError(
                    f"Channel drift in {store_path}: {name} has {list(pos.channel_names)}, expected {channel_names}"
                )
        fov_names = [name for name, _ in positions]

    channels = [
        ChannelEntry(
            name=ch,
            marker=target_cfg.gene if ch == target_cfg.target_channel else ch,
        )
        for ch in channel_names
    ]
    experiment = CollectionExperiment(
        name=experiment_name,
        data_path=store_path,
        channels=channels,
        organelle=target_cfg.organelle,
        marker=target_cfg.gene,
        pixel_size_xy_um=manifest_spacing.x,
        pixel_size_z_um=manifest_spacing.z,
    )
    prefixed = [f"{experiment_name}/{fov}" for fov in fov_names]
    return experiment, prefixed


def _default_description(dataset: str, target: str) -> str:
    """Write a short human-readable description for a single-pair collection."""
    return (
        f"Frozen benchmark collection for {dataset!r} target {target!r}. "
        "Channels and FOV membership captured from the live train/test "
        "zarr stores at freeze time."
    )
