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

from dynaclr.evaluation.paths import (
    DATASETS_ROOT,
    dataset_name_from_data_path,
    embedding_store,
)
from viscy_data.channel_utils import parse_channel_name
from viscy_data.collection import Collection, load_collection


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


def _slug(marker: str) -> str:
    """Filesystem-safe marker slug (kept readable, not lowercased)."""
    return marker


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


def _run_predict(
    entry: ReporterRun,
    *,
    checkpoint: str,
    encoder_kwargs: dict,
    example_input_array_shape: list[int],
    z_range: tuple[int, int],
    z_reduction: str | None,
    reference_pixel_size: float | None,
    yx_patch_size: tuple[int, int],
    batch_size: int,
    num_workers: int,
    uns_metadata: dict | None = None,
) -> None:
    """Load the checkpoint and run Lightning predict for one reporter."""
    import torch
    from lightning.pytorch import Trainer, seed_everything

    from dynaclr.engine import ContrastiveModule
    from viscy_data.triplet import TripletDataModule
    from viscy_models.contrastive import ContrastiveEncoder
    from viscy_transforms import NormalizeSampled
    from viscy_utils.callbacks.embedding_writer import EmbeddingWriter

    seed_everything(42)

    encoder = ContrastiveEncoder(**encoder_kwargs)
    module = ContrastiveModule(encoder=encoder, example_input_array_shape=example_input_array_shape)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    module.load_state_dict(ckpt["state_dict"])

    datamodule = TripletDataModule(
        data_path=entry.data_path,
        tracks_path=entry.tracks_path,
        source_channel=[entry.channel],
        z_range=list(z_range),
        z_reduction=z_reduction,
        reference_pixel_size=reference_pixel_size,
        initial_yx_patch_size=list(yx_patch_size),
        final_yx_patch_size=list(yx_patch_size),
        batch_size=batch_size,
        num_workers=num_workers,
        fit_include_wells=entry.wells,
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
    "--checkpoint",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Trained ContrastiveModule .ckpt.",
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
@click.option("--z-range", nargs=2, type=int, default=(15, 45), show_default=True, help="Z window (start stop).")
@click.option(
    "--z-reduction",
    type=click.Choice(["mip", "center"]),
    default="mip",
    show_default=True,
    help="Collapse z_range to one slice for a 2D model.",
)
@click.option(
    "--reference-pixel-size",
    type=float,
    default=0.1494,
    show_default=True,
    help="Model training pixel size (µm/px) for patch rescaling.",
)
@click.option("--yx-patch-size", nargs=2, type=int, default=(160, 160), show_default=True, help="Final YX patch.")
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--num-workers", type=int, default=0, show_default=True, help="Must be 0 for predict (zarr-fork).")
def main(
    collection_path: Path,
    checkpoint: Path,
    model_family: str,
    run: str,
    ckpt_name: str,
    datasets_root: Path,
    markers: str | None,
    include_labelfree: bool,
    z_range: tuple[int, int],
    z_reduction: str,
    reference_pixel_size: float,
    yx_patch_size: tuple[int, int],
    batch_size: int,
    num_workers: int,
) -> None:
    """Run per-reporter triplet embedding inference from a collection + checkpoint."""
    collection = load_collection(collection_path)
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
    click.echo(f"model_family : {model_family}")
    click.echo(f"run          : {run}")
    click.echo(f"checkpoint   : {ckpt_name} ({checkpoint})")
    click.echo(f"planned runs : {len(runs)}")
    for r in runs:
        wells = "all wells" if r.wells is None else ", ".join(r.wells)
        click.echo(f"  - {r.experiment} / {r.marker} ({r.channel}) [{wells}] -> {r.output_path}")

    example_input_array_shape = [1, 1, 1, yx_patch_size[0], yx_patch_size[1]]
    for r in runs:
        click.echo(f"\n=== Predicting: {r.experiment} / {r.marker} ===")
        provenance = {
            "model_family": model_family,
            "run": run,
            "ckpt_name": ckpt_name,
            "checkpoint": str(checkpoint),
            "collection_path": str(collection_path),
            "marker": r.marker,
            "channel": r.channel,
        }
        _run_predict(
            r,
            checkpoint=str(checkpoint),
            encoder_kwargs=_DEFAULT_ENCODER_KWARGS,
            example_input_array_shape=example_input_array_shape,
            z_range=z_range,
            z_reduction=z_reduction,
            reference_pixel_size=reference_pixel_size,
            yx_patch_size=yx_patch_size,
            batch_size=batch_size,
            num_workers=num_workers,
            uns_metadata=provenance,
        )
    click.echo(f"\nWrote {len(runs)} per-reporter embeddings zarrs.")


if __name__ == "__main__":
    main()
