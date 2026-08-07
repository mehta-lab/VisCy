"""One-call per-reporter triplet embedding inference.

Runs a trained ``ContrastiveModule`` checkpoint over an OME-Zarr + tracking
store (the triplet path; see ``docs/DAGs/inference_triplet.md``) and writes one
embeddings zarr per reporter. The collection YAML is the single source of truth:
each ``ChannelEntry`` carries a zarr ``name``, a ``marker`` label, and optional
``wells`` (empty = all wells). For bag-of-channels models (``in_channels=1``)
each channel is embedded as its own single-channel sample, so we run predict
once per channel, restricting to that channel's wells via ``fit_include_wells``.

Outputs are dataset/model/run/checkpoint-scoped so every embeddings zarr traces
back to what produced it, lives under its own dataset, and two models/checkpoints
can coexist (see :mod:`dynaclr.evaluation.paths`):

    {dataset}/2-phenotyping/predictions/{model_family}/{run}/{ckpt_name}/{marker}.zarr

This retires the per-dataset ``generate_predict_configs.py`` copies that caused
inference code to fragment across dataset folders.

Usage
-----
dynaclr predict-triplet \
    -c collection.yml \
    --checkpoint /path/to/epoch=105-step=84800.ckpt \
    --model-family DynaCLR-2D-MIP-BagOfChannels \
    --run 2d-mip-...-fix-shuffler \
    --ckpt-name epoch105-step84800 \
    --z-range 15 45 --z-reduction mip --reference-pixel-size 0.1494 \
    --no-labelfree            # skip Phase3D / brightfield channels
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import click
import yaml

from dynaclr.evaluation.paths import (
    DATASETS_ROOT,
    dataset_name_from_data_path,
    embedding_store,
)
from viscy_data.channel_utils import parse_channel_name
from viscy_data.collection import Collection, ExperimentEntry, load_collection

#: obs columns added by :func:`build_obs_metadata`, matching the parquet path's
#: cell-index schema so downstream steps (split-embeddings, MMD, LCs) consume
#: triplet embeddings without renaming. The triplet EmbeddingWriter otherwise
#: writes only ultrack index columns (fov_name/track_id/t/id/...).
COLLECTION_OBS_COLUMNS = (
    "experiment",
    "marker",
    "perturbation",
    "organelle",
    "microscope",
    "hours_post_perturbation",
    "interval_minutes",
)


@dataclass
class ReporterRun:
    """One per-reporter predict run derived from a collection channel.

    Parameters
    ----------
    experiment : str
        Experiment (dataset) name.
    marker : str
        Reporter/marker label used in the output filename.
    channel : str
        Zarr channel name fed as the single ``source_channel``.
    wells : list[str] | None
        Wells to restrict predict to (``fit_include_wells``). ``None`` means
        all wells (the channel is valid everywhere).
    data_path : str
        Resolved OME-Zarr store path.
    tracks_path : str
        Resolved tracking store path.
    pixel_size_xy_um : float | None
        Inference dataset pixel size, for the rescale log line.
    output_path : Path
        Destination embeddings zarr.
    is_labelfree : bool
        Whether the channel is label-free (phase/brightfield).
    """

    experiment: str
    marker: str
    channel: str
    wells: list[str] | None
    data_path: str
    tracks_path: str
    pixel_size_xy_um: float | None
    output_path: Path
    is_labelfree: bool
    exclude_fovs: list[str] | None = None


def _slug(marker: str) -> str:
    """Filesystem-safe marker slug (kept readable, not lowercased)."""
    return marker


def _well_of_fov(fov_name: str) -> str:
    """Well name (``row/col``) from an ultrack ``fov_name`` like ``A/2/000000``."""
    parts = str(fov_name).strip("/").split("/")
    return "/".join(parts[:2])


def _invert_perturbation_wells(perturbation_wells: dict[str, list[str]]) -> dict[str, str]:
    """Invert ``{label: [wells]}`` to ``{well: label}`` (last label wins on overlap)."""
    well_to_label: dict[str, str] = {}
    for label, wells in perturbation_wells.items():
        for well in wells:
            well_to_label[well] = label
    return well_to_label


def build_obs_metadata(obs, exp: ExperimentEntry, marker: str):
    """Derive collection metadata columns for an embedding's ultrack ``obs``.

    Pure, no-I/O: given the ultrack ``obs`` DataFrame written by the triplet
    ``EmbeddingWriter`` (columns include ``fov_name`` and ``t``) plus the
    collection experiment and the marker of this run, return a DataFrame —
    aligned to ``obs.index`` — carrying the same biological metadata columns the
    parquet path attaches (:data:`COLLECTION_OBS_COLUMNS`), so downstream steps
    consume triplet and parquet embeddings identically.

    ``perturbation`` is resolved per row from the cell's well
    (``fov_name`` → ``row/col``) via ``exp.perturbation_wells``; wells absent
    from the map resolve to ``"unknown"`` (matching the parquet builder).
    ``hours_post_perturbation`` uses the canonical
    ``start_hpi + t * interval_minutes / 60.0``.

    Parameters
    ----------
    obs : pandas.DataFrame
        Ultrack ``obs`` from the written embedding (needs ``fov_name``, ``t``).
    exp : ExperimentEntry
        Collection experiment carrying ``perturbation_wells``, ``organelle``,
        ``microscope``, ``interval_minutes``, ``start_hpi``.
    marker : str
        Marker/reporter label of this per-reporter run.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`COLLECTION_OBS_COLUMNS`, indexed like ``obs``.
    """
    import pandas as pd

    well_to_label = _invert_perturbation_wells(exp.perturbation_wells)
    wells = obs["fov_name"].map(_well_of_fov)
    perturbation = wells.map(lambda w: well_to_label.get(w, "unknown"))
    hpi = exp.start_hpi + obs["t"].astype(float) * exp.interval_minutes / 60.0

    return pd.DataFrame(
        {
            "experiment": exp.name,
            "marker": marker,
            "perturbation": perturbation.values,
            "organelle": exp.organelle,
            "microscope": exp.microscope,
            "hours_post_perturbation": hpi.values,
            "interval_minutes": exp.interval_minutes,
        },
        index=obs.index,
    )


def enrich_embedding_obs(zarr_path: Path, exp: ExperimentEntry, marker: str) -> dict[str, int]:
    """Add collection metadata columns to an already-written embedding zarr's obs.

    Reads the zarr's ``obs``, computes :func:`build_obs_metadata`, and writes the
    merged obs back with :func:`append_to_anndata_zarr` — which touches only the
    ``obs`` slot, leaving ``X``, ``obsm`` (X_backbone/X_projections), ``var`` and
    ``uns`` untouched. Existing obs columns are preserved; the collection columns
    are added (overwriting any prior copy of the same name).

    Parameters
    ----------
    zarr_path : Path
        The per-marker embedding zarr to enrich.
    exp : ExperimentEntry
        Collection experiment for this dataset.
    marker : str
        Marker/reporter label of this run.

    Returns
    -------
    dict[str, int]
        Per-perturbation-label cell counts (for logging / sanity checks).
    """
    import anndata as ad

    from viscy_utils.evaluation.zarr_utils import append_to_anndata_zarr

    adata = ad.read_zarr(zarr_path)
    meta = build_obs_metadata(adata.obs, exp, marker)
    for col in COLLECTION_OBS_COLUMNS:
        adata.obs[col] = meta[col].values
    append_to_anndata_zarr(zarr_path, obs=adata.obs)
    return adata.obs["perturbation"].value_counts().to_dict()


def plan_predict_runs(
    collection: Collection,
    *,
    model_family: str,
    run: str,
    ckpt_name: str,
    datasets_root: str | Path = DATASETS_ROOT,
    markers: list[str] | None = None,
    include_labelfree: bool = True,
) -> list[ReporterRun]:
    """Build the list of per-reporter predict runs from a collection.

    Pure planning logic (no I/O beyond path construction) so it can be unit
    tested without a GPU or real zarr.

    Each embedding zarr lands in the dataset-centric tree
    ``{dataset}/2-phenotyping/predictions/{model_family}/{run}/{ckpt_name}/{marker}.zarr``
    (see :mod:`dynaclr.evaluation.paths`). The dataset is derived per experiment
    from its ``data_path``, so multiple markers of one physical dataset
    co-locate in the same ``{model_family}/{run}/{ckpt_name}/`` directory.

    Parameters
    ----------
    collection : Collection
        Loaded collection (``${datasets_root}`` already resolved).
    model_family, run, ckpt_name : str
        Provenance identity, keyed into the output path.
    datasets_root : str or Path, optional
        Base under which datasets live. Defaults to
        :data:`dynaclr.evaluation.paths.DATASETS_ROOT`; each experiment's
        dataset folder is derived from its ``data_path``, so this only matters
        if that folder is not already under the canonical root.
    markers : list[str] | None
        If given, only emit runs for these marker labels. ``None`` = all.
    include_labelfree : bool
        If ``False``, skip label-free channels (phase/brightfield), resolved
        by name via :func:`parse_channel_name`.

    Returns
    -------
    list[ReporterRun]
        One entry per (experiment, channel) to embed.
    """
    runs: list[ReporterRun] = []
    for exp in collection.experiments:
        dataset = dataset_name_from_data_path(exp.data_path)
        for ch in exp.channels:
            is_labelfree = parse_channel_name(ch.name)["channel_type"] == "labelfree"
            if not include_labelfree and is_labelfree:
                continue
            if markers is not None and ch.marker not in markers:
                continue
            runs.append(
                ReporterRun(
                    experiment=exp.name,
                    marker=ch.marker,
                    channel=ch.name,
                    wells=list(ch.wells) if ch.wells else None,
                    exclude_fovs=list(exp.exclude_fovs) if exp.exclude_fovs else None,
                    data_path=exp.data_path,
                    tracks_path=exp.tracks_path,
                    pixel_size_xy_um=exp.pixel_size_xy_um,
                    output_path=embedding_store(
                        dataset,
                        model_family,
                        run,
                        ckpt_name,
                        _slug(ch.marker),
                        datasets_root=datasets_root,
                    ),
                    is_labelfree=is_labelfree,
                )
            )
    if not runs:
        raise ValueError(
            "No reporter runs planned. Check that the collection has channels "
            "matching --markers / --no-labelfree filters."
        )
    return runs


def _build_dynaclr_module(checkpoint: str, encoder_kwargs: dict, example_input_array_shape: list[int]):
    """Construct a ``ContrastiveModule`` from encoder kwargs + a Lightning checkpoint."""
    import torch

    from dynaclr.engine import ContrastiveModule
    from viscy_models.contrastive import ContrastiveEncoder

    encoder = ContrastiveEncoder(**encoder_kwargs)
    module = ContrastiveModule(encoder=encoder, example_input_array_shape=example_input_array_shape)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    module.load_state_dict(ckpt["state_dict"])
    return module


def _build_foundation_module(training_config: str):
    """Instantiate the ``model:`` block of a LightningCLI training_config YAML.

    Foundation baselines (CellDino/DINOv3/MorphEm) are configured as a
    ``ContrastiveModule`` whose ``encoder`` is a foundation wrapper. There is no
    Lightning checkpoint — the wrapper loads its own weights (HF download or a
    local ``.pth``) in ``__init__``. We reuse LightningCLI's own
    class_path/init_args instantiator (``LightningArgumentParser`` +
    ``instantiate_classes``) so the ``model:`` block is built exactly as
    ``dynaclr fit`` would build it.
    """
    from lightning.pytorch.cli import LightningArgumentParser

    from dynaclr.engine import ContrastiveModule

    config = yaml.safe_load(Path(training_config).read_text())
    parser = LightningArgumentParser()
    parser.add_lightning_class_args(ContrastiveModule, "model")
    args = parser.parse_object({"model": config["model"]["init_args"]})
    init = parser.instantiate_classes(args)
    return init.model


def _run_predict(
    entry: ReporterRun,
    *,
    model_type: str,
    checkpoint: str | None,
    training_config: str | None,
    encoder_kwargs: dict,
    example_input_array_shape: list[int],
    z_range: tuple[int, int] | None,
    z_extraction_window: int | None,
    focus_channel: str | None,
    z_focus_offset: float,
    z_reduction: str | None,
    reference_pixel_size: float | None,
    reference_pixel_size_z_um: float | None,
    yx_patch_size: tuple[int, int],
    batch_size: int,
    num_workers: int,
    uns_metadata: dict | None = None,
) -> None:
    """Build the model and run Lightning predict for one reporter.

    ``model_type="dynaclr"`` builds a :class:`ContrastiveModule` from
    ``encoder_kwargs`` and loads ``checkpoint``. ``model_type="foundation"``
    instantiates the ``model:`` block of ``training_config`` (weights load inside
    the foundation wrapper — no Lightning checkpoint).

    Z selection is either a fixed absolute ``z_range`` (same slices for every
    FOV) or, when ``z_extraction_window`` is given, a fixed-width window centered
    per-FOV on that FOV's focus plane (``focus_channel`` / ``z_focus_offset``).
    The two are mutually exclusive — ``TripletDataModule`` enforces this.
    """
    from lightning.pytorch import Trainer, seed_everything

    from viscy_data.triplet import TripletDataModule
    from viscy_transforms import NormalizeSampled
    from viscy_utils.callbacks.embedding_writer import EmbeddingWriter

    seed_everything(42)

    if model_type == "dynaclr":
        module = _build_dynaclr_module(checkpoint, encoder_kwargs, example_input_array_shape)
    elif model_type == "foundation":
        module = _build_foundation_module(training_config)
    else:
        raise ValueError(f"unknown model_type {model_type!r}; expected 'dynaclr' or 'foundation'")

    # Fixed absolute window vs per-FOV focus-centered window (mutually exclusive).
    if z_extraction_window is not None:
        z_kwargs = {
            "z_extraction_window": z_extraction_window,
            "focus_channel": focus_channel,
            "z_focus_offset": z_focus_offset,
        }
    else:
        z_kwargs = {"z_range": list(z_range)}

    datamodule = TripletDataModule(
        data_path=entry.data_path,
        tracks_path=entry.tracks_path,
        source_channel=[entry.channel],
        **z_kwargs,
        z_reduction=z_reduction,
        reference_pixel_size=reference_pixel_size,
        reference_pixel_size_z_um=reference_pixel_size_z_um,
        initial_yx_patch_size=list(yx_patch_size),
        final_yx_patch_size=list(yx_patch_size),
        batch_size=batch_size,
        num_workers=num_workers,
        fit_include_wells=entry.wells,
        fit_exclude_fovs=entry.exclude_fovs,
        normalizations=[
            NormalizeSampled(
                keys=[entry.channel],
                level="fov_statistics",
                subtrahend="mean",
                divisor="std",
            )
        ],
    )

    writer = EmbeddingWriter(
        output_path=entry.output_path,
        embedding_key="features",
        overwrite=True,
        pca_kwargs=None,
        phate_kwargs=None,
        umap_kwargs=None,
        uns_metadata=uns_metadata,
    )
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        precision="32-true",
        callbacks=[writer],
        inference_mode=True,
        logger=False,
    )
    entry.output_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.predict(module, datamodule=datamodule, return_predictions=False)


# ConvNeXt-tiny bag-of-channels encoder — matches DynaCLR-2D-MIP-BagOfChannels.
_DEFAULT_ENCODER_KWARGS = {
    "backbone": "convnext_tiny",
    "in_channels": 1,
    "in_stack_depth": 1,
    "stem_kernel_size": [1, 4, 4],
    "stem_stride": [1, 4, 4],
    "embedding_dim": 768,
    "projection_dim": 32,
    "drop_path_rate": 0.0,
}


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-c",
    "--collection",
    "collection_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Collection YAML (experiments with channels: name/marker/wells).",
)
@click.option(
    "--model-type",
    type=click.Choice(["dynaclr", "foundation"]),
    default="dynaclr",
    show_default=True,
    help="'dynaclr' loads a checkpoint into a ContrastiveEncoder; 'foundation' "
    "instantiates the model: block of --training-config (weights in the wrapper).",
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Trained ContrastiveModule .ckpt (required for --model-type dynaclr).",
)
@click.option(
    "--training-config",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="LightningCLI training_config YAML whose model: block builds the "
    "foundation ContrastiveModule (required for --model-type foundation).",
)
@click.option("--model-family", required=True, help="Model family (top output folder).")
@click.option("--run", required=True, help="Training run/version (second output folder).")
@click.option("--ckpt-name", required=True, help="Checkpoint label, e.g. epoch105-step84800.")
@click.option(
    "--datasets-root",
    type=click.Path(path_type=Path),
    default=DATASETS_ROOT,
    show_default=True,
    help="Base under which datasets live; each embedding lands in "
    "<dataset>/2-phenotyping/predictions/{model_family}/{run}/{ckpt_name}/{marker}.zarr.",
)
@click.option(
    "--markers",
    default=None,
    help="Comma-separated marker subset to embed (default: all channels).",
)
@click.option(
    "--no-labelfree",
    "include_labelfree",
    is_flag=True,
    default=True,
    flag_value=False,
    help="Skip label-free (phase/brightfield) channels.",
)
@click.option(
    "--z-range",
    nargs=2,
    type=int,
    default=None,
    help="Fixed absolute Z window (start stop), same slices for every FOV. "
    "Mutually exclusive with --z-window. Default (neither given): (15, 45).",
)
@click.option(
    "--z-window",
    type=int,
    default=None,
    help="Focus-centered Z window WIDTH (slices) centered per-FOV on the focus plane "
    "(from focus_slice zattrs). Mutually exclusive with --z-range.",
)
@click.option(
    "--focus-channel",
    default=None,
    help="Channel whose focus_slice metadata centers the --z-window per FOV (e.g. Phase3D).",
)
@click.option(
    "--z-focus-offset",
    type=float,
    default=0.3,
    show_default=True,
    help="Fraction of --z-window placed below the focus plane (0.5 = symmetric).",
)
@click.option(
    "--z-reduction",
    type=click.Choice(["mip", "center"]),
    default="mip",
    show_default=True,
    help="Collapse the Z window to one slice for a 2D model.",
)
@click.option(
    "--reference-pixel-size",
    type=float,
    default=0.1494,
    show_default=True,
    help="Model training pixel size (µm/px) for patch rescaling.",
)
@click.option(
    "--reference-pixel-size-z-um",
    type=float,
    default=None,
    help="Model training Z sampling (µm/slice). With --z-window, converts its "
    "reference-grid slice count to the native count covering the same physical depth.",
)
@click.option("--yx-patch-size", nargs=2, type=int, default=(160, 160), show_default=True, help="Final YX patch.")
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--num-workers", type=int, default=0, show_default=True, help="Must be 0 for predict (zarr-fork).")
@click.option(
    "--no-enrich-obs",
    "enrich_obs",
    is_flag=True,
    default=True,
    flag_value=False,
    help="Skip adding collection metadata (perturbation/hpi/...) to obs.",
)
def main(
    collection_path: Path,
    model_type: str,
    checkpoint: Path | None,
    training_config: Path | None,
    model_family: str,
    run: str,
    ckpt_name: str,
    datasets_root: Path,
    markers: str | None,
    include_labelfree: bool,
    z_range: tuple[int, int] | None,
    z_window: int | None,
    focus_channel: str | None,
    z_focus_offset: float,
    z_reduction: str,
    reference_pixel_size: float,
    reference_pixel_size_z_um: float | None,
    yx_patch_size: tuple[int, int],
    batch_size: int,
    num_workers: int,
    enrich_obs: bool,
) -> None:
    """Run per-reporter triplet embedding inference from a collection + model."""
    if model_type == "dynaclr" and checkpoint is None:
        raise click.UsageError("--checkpoint is required for --model-type dynaclr.")
    if model_type == "foundation" and training_config is None:
        raise click.UsageError("--training-config is required for --model-type foundation.")
    if z_range and z_window is not None:
        raise click.UsageError("--z-range and --z-window are mutually exclusive; pass only one.")
    if reference_pixel_size_z_um is not None and z_window is None:
        raise click.UsageError("--reference-pixel-size-z-um requires --z-window (focus-centered extraction).")
    # Focus-centered when --z-window given; otherwise fixed window (default (15, 45)).
    if z_window is None and not z_range:
        z_range = (15, 45)

    collection = load_collection(collection_path)
    experiments_by_name = {exp.name: exp for exp in collection.experiments}
    marker_list = [m.strip() for m in markers.split(",")] if markers else None
    runs = plan_predict_runs(
        collection,
        model_family=model_family,
        run=run,
        ckpt_name=ckpt_name,
        datasets_root=datasets_root,
        markers=marker_list,
        include_labelfree=include_labelfree,
    )

    click.echo("=== Provenance ===")
    click.echo(f"model_type   : {model_type}")
    click.echo(f"model_family : {model_family}")
    click.echo(f"run          : {run}")
    if model_type == "foundation":
        click.echo(f"training_cfg : {ckpt_name} ({training_config})")
    else:
        click.echo(f"checkpoint   : {ckpt_name} ({checkpoint})")
    click.echo(f"planned runs : {len(runs)}")
    for r in runs:
        wells = "all wells" if r.wells is None else ", ".join(r.wells)
        click.echo(f"  - {r.experiment} / {r.marker} ({r.channel}) [{wells}] -> {r.output_path}")

    example_input_array_shape = [1, 1, 1, yx_patch_size[0], yx_patch_size[1]]
    for r in runs:
        click.echo(f"\n=== Predicting: {r.experiment} / {r.marker} ===")
        provenance = {
            "model_type": model_type,
            "model_family": model_family,
            "run": run,
            "ckpt_name": ckpt_name,
            "checkpoint": str(checkpoint) if checkpoint else "",
            "training_config": str(training_config) if training_config else "",
            "collection_path": str(collection_path),
            "marker": r.marker,
            "channel": r.channel,
            "reference_pixel_size_z_um": reference_pixel_size_z_um,
        }
        _run_predict(
            r,
            model_type=model_type,
            checkpoint=str(checkpoint) if checkpoint else None,
            training_config=str(training_config) if training_config else None,
            encoder_kwargs=_DEFAULT_ENCODER_KWARGS,
            example_input_array_shape=example_input_array_shape,
            z_range=z_range,
            z_extraction_window=z_window,
            focus_channel=focus_channel,
            z_focus_offset=z_focus_offset,
            z_reduction=z_reduction,
            reference_pixel_size=reference_pixel_size,
            reference_pixel_size_z_um=reference_pixel_size_z_um,
            yx_patch_size=yx_patch_size,
            batch_size=batch_size,
            num_workers=num_workers,
            uns_metadata=provenance,
        )
        if enrich_obs:
            counts = enrich_embedding_obs(r.output_path, experiments_by_name[r.experiment], r.marker)
            click.echo(f"  enriched obs ({', '.join(COLLECTION_OBS_COLUMNS)}); perturbation counts: {counts}")
    click.echo(f"\nWrote {len(runs)} per-reporter embeddings zarrs.")


@click.command("enrich-obs-from-collection", context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-c",
    "--collection",
    "collection_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Collection YAML the embeddings were produced from.",
)
@click.option("--model-family", required=True, help="Model family (as used at predict time).")
@click.option("--run", required=True, help="Training run/version.")
@click.option("--ckpt-name", required=True, help="Checkpoint label, e.g. epoch105-step84800.")
@click.option(
    "--datasets-root",
    type=click.Path(path_type=Path),
    default=DATASETS_ROOT,
    show_default=True,
    help="Base under which datasets live.",
)
@click.option("--markers", default=None, help="Comma-separated marker subset (default: all channels).")
@click.option(
    "--no-labelfree",
    "include_labelfree",
    is_flag=True,
    default=True,
    flag_value=False,
    help="Skip label-free channels.",
)
def enrich_main(
    collection_path: Path,
    model_family: str,
    run: str,
    ckpt_name: str,
    datasets_root: Path,
    markers: str | None,
    include_labelfree: bool,
) -> None:
    """Backfill collection metadata onto already-produced per-marker embedding zarrs.

    Adds perturbation / hours_post_perturbation / experiment / marker / organelle
    / microscope / interval_minutes to each embedding's obs (obsm/X untouched),
    for embeddings that were written before obs enrichment existed. Idempotent —
    re-running overwrites the same columns.
    """
    collection = load_collection(collection_path)
    experiments_by_name = {exp.name: exp for exp in collection.experiments}
    marker_list = [m.strip() for m in markers.split(",")] if markers else None
    runs = plan_predict_runs(
        collection,
        model_family=model_family,
        run=run,
        ckpt_name=ckpt_name,
        datasets_root=datasets_root,
        markers=marker_list,
        include_labelfree=include_labelfree,
    )
    enriched = 0
    for r in runs:
        if not r.output_path.exists():
            click.echo(f"  SKIP (missing): {r.output_path}")
            continue
        counts = enrich_embedding_obs(r.output_path, experiments_by_name[r.experiment], r.marker)
        click.echo(f"  {r.experiment} / {r.marker}: perturbation counts {counts} -> {r.output_path}")
        enriched += 1
    click.echo(f"\nEnriched {enriched}/{len(runs)} embedding zarrs.")


if __name__ == "__main__":
    main()
