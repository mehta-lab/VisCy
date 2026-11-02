r"""Batch predict embeddings for one model over a collection's datasets.

Wraps ``dynaclr predict-triplet`` — the predict stage of the model matrix. Given
a model identity ``(model_family, run, ckpt_name)`` and a collection, it builds
the checkpoint path (from the training run dir) and the ``predict-triplet``
command that writes per-marker embeddings into the dataset-centric tree
``<dataset>/2-phenotyping/predictions/{model_family}/{run}/{ckpt_name}/{marker}.zarr``.

Used standalone and by ``run-matrix`` (the predict link of the chain).

Checkpoint selection is manual via ``--ckpt-name``:
  * ``last``               → ``<run_dir>/checkpoints/last.ckpt``
  * ``epoch105-step84800`` → ``<run_dir>/checkpoints/epoch=105-step=84800.ckpt``

Output:
  * ``--print-cmd``: print the ``dynaclr predict-triplet`` command (one token per line).
  * default: execute it via ``subprocess.run``.

Usage::

    dynaclr predict-batch \
        -c applications/dynaclr/configs/collections/<...>/<dataset>.yml \
        --model-family DynaCLR-2D-MIP-BagOfChannels \
        --run 2d-mip-fix-shuffler --ckpt-name last \
        --datasets-root /hpc/projects/intracellular_dashboard/organelle_dynamics \
        [--markers SEC61B] [--no-labelfree] [--print-cmd]
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import click

from dynaclr.evaluation.paths import DATASETS_ROOT

#: Default training-model root (where {model_family}/{run}/checkpoints/ live).
MODELS_ROOT = Path("/hpc/projects/organelle_phenotyping/models")

#: Canonical reconstruction workspace used to populate the VAST AI-ready stores.
SOURCE_DATASETS_ROOT = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics")


def _zarr_metadata_mtime_ns(path: str | Path) -> int:
    """Return the root metadata timestamp for a Zarr v2 or v3 store."""
    path = Path(path)
    for filename in ("zarr.json", ".zattrs", ".zgroup"):
        metadata = path / filename
        if metadata.exists():
            return metadata.stat().st_mtime_ns
    return path.stat().st_mtime_ns


def _latest_source_assembly(dataset_name: str, source_root: Path = SOURCE_DATASETS_ROOT) -> Path | None:
    """Find the newest active assembled store for one dashboard dataset.

    Pipeline generations use different numbered stage names (for example,
    ``2-assemble``, ``4-assemble``, and ``5-assemble``).  Restricting the glob
    to a direct child of those stage directories intentionally excludes
    archived runs such as ``prev_run_tfbug`` and ``.iohub-progress`` stores.
    """
    dataset_dir = source_root / dataset_name
    candidates = [
        path
        for pattern in ("*-assemble/*.zarr", "*-assembly/*.zarr")
        for path in dataset_dir.glob(pattern)
        if path.is_dir()
    ]
    if not candidates:
        return None
    return max(candidates, key=_zarr_metadata_mtime_ns)


def _format_mtime(mtime_ns: int | None) -> str | None:
    if mtime_ns is None:
        return None
    return datetime.fromtimestamp(mtime_ns / 1e9).astimezone().isoformat(timespec="seconds")


def check_ai_ready(
    collection: Path,
    *,
    source_root: Path = SOURCE_DATASETS_ROOT,
) -> list[dict]:
    """Report which of a collection's datasets are not AI-ready.

    ``predict-triplet`` reads ``normalization`` and ``focus_slice`` from each
    FOV's zattrs (written upstream). This inspects the first FOV of every
    experiment's ``data_path`` and returns, per dataset, which of the two are
    missing — so the caller can flag them (focus_slice needs manual z-focus
    physics params, so it is never auto-run) or auto-run the safe normalization.

    Parameters
    ----------
    collection : Path
        Collection YAML; ``${datasets_root}`` is resolved by ``load_collection``.

    Returns
    -------
    list[dict]
        One entry per experiment. In addition to normalization/focus flags,
        dashboard-backed datasets include the inferred source assembly and a
        ``stale_source`` flag comparing its root metadata timestamp with VAST.
    """
    from iohub import open_ome_zarr

    from viscy_data.collection import load_collection

    coll = load_collection(collection)
    report: list[dict] = []
    for exp in coll.experiments:
        with open_ome_zarr(exp.data_path, mode="r") as plate:
            _, pos = next(plate.positions())
            attrs = dict(pos.zattrs)
        required_normalization = list(dict.fromkeys(channel.name for channel in exp.channels))
        required_focus = [channel for channel in required_normalization if channel == "Phase3D"]
        normalization = attrs.get("normalization", {})
        focus_slice = attrs.get("focus_slice", {})
        missing_normalization_channels = [channel for channel in required_normalization if channel not in normalization]
        missing_focus_slice_channels = [channel for channel in required_focus if channel not in focus_slice]
        source_assembly = _latest_source_assembly(exp.name, source_root)
        vast_mtime_ns = _zarr_metadata_mtime_ns(exp.data_path)
        source_mtime_ns = _zarr_metadata_mtime_ns(source_assembly) if source_assembly is not None else None
        report.append(
            {
                "name": exp.name,
                "data_path": exp.data_path,
                "missing_normalization": bool(missing_normalization_channels),
                "missing_normalization_channels": missing_normalization_channels,
                "missing_focus_slice": bool(missing_focus_slice_channels),
                "missing_focus_slice_channels": missing_focus_slice_channels,
                "source_assembly": str(source_assembly) if source_assembly is not None else None,
                "source_mtime": _format_mtime(source_mtime_ns),
                "vast_mtime": _format_mtime(vast_mtime_ns),
                "stale_source": source_mtime_ns is not None and source_mtime_ns > vast_mtime_ns,
            }
        )
    return report


def run_normalization(
    data_path: str,
    workspace_dir: str,
    channel_names: list[str],
    num_workers: int = 32,
    block_size: int = 32,
) -> None:
    """Auto-run ``viscy preprocess`` for the required missing channels.

    Safe to run unattended — it takes no dataset-specific physics params
    (unlike QC/focus_slice). Writes ``normalization`` zattrs into ``data_path``.
    """
    cmd = [
        "uv",
        "run",
        "--project",
        workspace_dir,
        "--package",
        "dynaclr",
        "viscy",
        "preprocess",
        "--data_path",
        data_path,
    ]
    cmd += [
        "--channel_names",
        json.dumps(channel_names),
        "--num_workers",
        str(num_workers),
        "--block_size",
        str(block_size),
    ]
    print(f"  auto-running normalization: {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)


def preflight(
    collection: Path,
    workspace_dir: str,
    auto_normalize: bool,
    *,
    source_root: Path = SOURCE_DATASETS_ROOT,
) -> None:
    """Report non-AI-ready datasets; optionally auto-run normalization.

    focus_slice is only **flagged** (never auto-run — it needs manual z-focus
    params). If any dataset is missing focus_slice, raise so the user runs QC
    themselves (``qc run -c qc_config.yml``).
    """
    report = check_ai_ready(collection, source_root=source_root)
    stale_sources = [r for r in report if r["stale_source"]]
    needs_focus = [r for r in report if r["missing_focus_slice"]]
    needs_norm = [r for r in report if r["missing_normalization"]]

    if stale_sources:
        lines = "\n".join(
            f"    - {r['name']}: source {r['source_mtime']} ({r['source_assembly']}) "
            f"> VAST {r['vast_mtime']} ({r['data_path']})"
            for r in stale_sources
        )
        raise ValueError(
            "Refusing inference because these VAST datasets are older than their "
            "active source assemblies:\n"
            f"{lines}\n"
            "Re-run the AI-ready preparation pipeline from the corrected assembly, "
            "including QC and preprocessing, then retry."
        )

    for r in needs_norm:
        if auto_normalize:
            run_normalization(
                r["data_path"],
                workspace_dir,
                r["missing_normalization_channels"],
            )
        else:
            channels = ", ".join(r["missing_normalization_channels"])
            print(
                f"  [flag] {r['name']}: missing normalization for [{channels}] ({r['data_path']})",
                file=sys.stderr,
            )

    if needs_focus:
        lines = "\n".join(
            f"    - {r['name']}: {r['missing_focus_slice_channels']} ({r['data_path']})" for r in needs_focus
        )
        raise ValueError(
            "These datasets are missing focus_slice zattrs and need QC run manually "
            "(focus finding needs per-dataset z-focus physics params — NA, wavelength, "
            "pixel size — so it is not auto-run):\n"
            f"{lines}\n"
            "Run `qc run -c <dataset>/qc_config.yml` for each, then retry."
        )
    if needs_norm and not auto_normalize:
        raise ValueError(
            "Datasets missing required normalization channel metadata (rerun with "
            "--auto-normalize to compute only those channels, or run `viscy preprocess "
            "--data_path <zarr> --channel_names <required-channel>` yourself)."
        )


def checkpoint_path(
    model_family: str,
    run: str,
    ckpt_name: str,
    models_root: str | Path = MODELS_ROOT,
) -> Path:
    """Resolve the checkpoint file from the model identity + manual label.

    Parameters
    ----------
    model_family, run : str
        Training ``PROJECT`` / ``RUN_NAME`` — the run dir is
        ``{models_root}/{model_family}/{run}/checkpoints``.
    ckpt_name : str
        ``last`` → ``last.ckpt``; otherwise a ``epochN-stepM`` label mapped to
        ``epoch=N-step=M.ckpt``.
    models_root : str or Path, optional
        Base under which trained models live. Defaults to :data:`MODELS_ROOT`.

    Returns
    -------
    Path
        The checkpoint file path.
    """
    ckpt_dir = Path(models_root) / model_family / run / "checkpoints"
    if ckpt_name == "last":
        filename = "last.ckpt"
    else:
        # epoch105-step84800 -> epoch=105-step=84800.ckpt
        filename = ckpt_name.replace("epoch", "epoch=").replace("-step", "-step=") + ".ckpt"
    return ckpt_dir / filename


def build_predict_cmd(
    collection: Path,
    checkpoint: Path | None,
    model_family: str,
    run: str,
    ckpt_name: str,
    datasets_root: str | Path,
    predict_flags: dict | None = None,
    markers: list[str] | None = None,
    no_labelfree: bool = False,
    num_workers: int = 0,
    model_type: str = "dynaclr",
    training_config: str | Path | None = None,
    eligibility: str | Path | None = None,
) -> list[str]:
    """Build the ``dynaclr predict-triplet`` command.

    ``model_type="foundation"`` forwards ``--training-config`` (the LightningCLI
    model config that builds the frozen foundation module) and omits
    ``--checkpoint``; ``model_type="dynaclr"`` forwards ``--checkpoint``.
    """
    from dynaclr.evaluation.phenotyping_eligibility import resolve_eligibility_path

    eligibility_path = resolve_eligibility_path(eligibility, predict_flags)
    if model_type == "katamari":
        cmd = [sys.executable, "-m", "dynaclr.evaluation.predict_triplet"]
    else:
        cmd = ["dynaclr", "predict-triplet"]
    cmd += [
        "-c",
        str(collection),
        "--model-type",
        model_type,
        "--model-family",
        model_family,
        "--run",
        run,
        "--ckpt-name",
        ckpt_name,
        "--datasets-root",
        str(datasets_root),
        "--num-workers",
        str(num_workers),
    ]
    if model_type == "foundation":
        cmd += ["--training-config", str(training_config)]
    else:
        cmd += ["--checkpoint", str(checkpoint)]
    if eligibility_path is not None:
        cmd += ["--eligibility", str(eligibility_path)]
    flags = predict_flags or {}
    if "z_range" in flags:
        cmd += ["--z-range", str(flags["z_range"][0]), str(flags["z_range"][1])]
    if "z_reduction" in flags:
        cmd += ["--z-reduction", str(flags["z_reduction"])]
    if "reference_pixel_size" in flags:
        cmd += ["--reference-pixel-size", str(flags["reference_pixel_size"])]
    if "reference_pixel_size_z_um" in flags:
        cmd += ["--reference-pixel-size-z-um", str(flags["reference_pixel_size_z_um"])]
    if "batch_size" in flags:
        cmd += ["--batch-size", str(flags["batch_size"])]
    if "z_window" in flags:
        cmd += ["--z-window", str(flags["z_window"])]
    if "focus_channel" in flags:
        cmd += ["--focus-channel", str(flags["focus_channel"])]
    if "z_focus_offset" in flags:
        cmd += ["--z-focus-offset", str(flags["z_focus_offset"])]
    if "yx_patch_size" in flags:
        cmd += ["--yx-patch-size", str(flags["yx_patch_size"][0]), str(flags["yx_patch_size"][1])]
    if "katamari_model_input_size" in flags:
        size = flags["katamari_model_input_size"]
        cmd += ["--katamari-model-input-size", str(size[0]), str(size[1])]
    if "katamari_readout" in flags:
        cmd += ["--katamari-readout", str(flags["katamari_readout"])]
    if flags.get("katamari_use_teacher"):
        cmd.append("--katamari-use-teacher")
    if markers:
        cmd += ["--markers", ",".join(markers)]
    if no_labelfree:
        cmd.append("--no-labelfree")
    return cmd


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-c",
    "--collection",
    type=click.Path(path_type=Path),
    required=True,
    help="Collection YAML (predict-triplet input).",
)
@click.option(
    "--model-type",
    type=click.Choice(["dynaclr", "foundation", "katamari"]),
    default="dynaclr",
    show_default=True,
    help="'foundation' forwards --training-config; 'katamari' uses an explicit checkpoint.",
)
@click.option(
    "--training-config",
    type=click.Path(path_type=Path),
    default=None,
    help="LightningCLI training_config YAML for --model-type foundation.",
)
@click.option("--model-family", required=True, help="Model family (PROJECT / embedding-tree key).")
@click.option("--run", required=True, help="Training run/version (RUN_NAME / embedding-tree key).")
@click.option("--ckpt-name", default="last", help="Checkpoint label: 'last' or 'epochN-stepM' (default: last).")
@click.option(
    "--checkpoint",
    type=click.Path(path_type=Path),
    default=None,
    help="Explicit checkpoint path — overrides derivation from --models-root/--run/--ckpt-name. "
    "Use when the checkpoint is under a wandb-run-id subdir (Lightning's default nesting).",
)
@click.option(
    "--models-root",
    type=click.Path(path_type=Path),
    default=MODELS_ROOT,
    show_default=True,
    help="Base under which trained models live.",
)
@click.option(
    "--datasets-root",
    type=click.Path(path_type=Path),
    default=DATASETS_ROOT,
    show_default=True,
    help="Base under which datasets live.",
)
@click.option("--z-range", nargs=2, type=int, default=None, help="Fixed Z window forwarded to predict-triplet.")
@click.option("--z-window", type=int, default=None, help="Focus-centered Z window WIDTH forwarded to predict-triplet.")
@click.option("--focus-channel", default=None, help="Focus channel for --z-window, forwarded to predict-triplet.")
@click.option("--z-focus-offset", type=float, default=None, help="Focus offset for --z-window, forwarded.")
@click.option(
    "--z-reduction",
    type=click.Choice(["mip", "center"]),
    default=None,
    help="Z reduction forwarded to predict-triplet.",
)
@click.option(
    "--reference-pixel-size",
    type=float,
    default=None,
    help="Reference pixel size forwarded to predict-triplet.",
)
@click.option(
    "--reference-pixel-size-z-um",
    type=float,
    default=None,
    help="Reference Z sampling forwarded to predict-triplet.",
)
@click.option("--batch-size", type=int, default=None, help="Batch size forwarded to predict-triplet.")
@click.option("--yx-patch-size", nargs=2, type=int, default=None, help="Final YX patch forwarded to predict-triplet.")
@click.option(
    "--katamari-model-input-size",
    nargs=2,
    type=int,
    default=None,
    help="Katamari encoder YX size forwarded to predict-triplet.",
)
@click.option("--katamari-readout", type=click.Choice(["avg", "cls", "cls_plus_gap"]), default=None)
@click.option("--katamari-use-teacher", is_flag=True, help="Use Katamari EMA-teacher weights instead of the student.")
@click.option("--markers", multiple=True, default=None, help="Marker subset (default: all channels).")
@click.option("--no-labelfree", is_flag=True, help="Skip label-free (phase/brightfield) channels.")
@click.option("--num-workers", type=int, default=0, help="Predict dataloader workers (must be 0).")
@click.option(
    "--eligibility",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional approved eligibility parquet, forwarded to predict-triplet before image patches are read.",
)
@click.option("--print-cmd", is_flag=True, help="Print the command instead of executing it.")
@click.option(
    "--auto-normalize",
    is_flag=True,
    help="Auto-run `viscy preprocess` for datasets missing normalization zattrs (safe, no manual params). "
    "focus_slice is always only flagged, never auto-run.",
)
@click.option(
    "--skip-preflight",
    is_flag=True,
    help="Skip the AI-ready (normalization/focus_slice zattrs) preflight entirely.",
)
@click.option(
    "--workspace-dir",
    default="/hpc/mydata/eduardo.hirata/repos/viscy",
    help="uv workspace dir (for the auto-normalize preprocess call).",
)
def main(
    collection: Path,
    model_type: str,
    training_config: Path | None,
    model_family: str,
    run: str,
    ckpt_name: str,
    checkpoint: Path | None,
    models_root: Path,
    datasets_root: Path,
    z_range: tuple[int, int] | None,
    z_reduction: str | None,
    reference_pixel_size: float | None,
    reference_pixel_size_z_um: float | None,
    batch_size: int | None,
    yx_patch_size: tuple[int, int] | None,
    katamari_model_input_size: tuple[int, int] | None,
    katamari_readout: str | None,
    katamari_use_teacher: bool,
    z_window: int | None,
    focus_channel: str | None,
    z_focus_offset: float | None,
    markers: tuple[str, ...],
    no_labelfree: bool,
    num_workers: int,
    print_cmd: bool,
    auto_normalize: bool,
    skip_preflight: bool,
    workspace_dir: str,
    eligibility: Path | None = None,
) -> None:
    """Batch predict embeddings for one model over a collection (+ AI-ready preflight).

    ``--markers`` runs exactly the named markers (extras in the collection are
    ignored); omit it to embed all channels. ``--model-type foundation`` forwards
    ``--training-config`` and needs no checkpoint.
    """
    if not print_cmd and not skip_preflight:
        preflight(collection, workspace_dir, auto_normalize=auto_normalize)

    if model_type == "foundation":
        if training_config is None:
            raise click.UsageError("--training-config is required for --model-type foundation.")
        resolved_checkpoint = None
    elif model_type == "katamari":
        if checkpoint is None:
            raise click.UsageError("--checkpoint is required for --model-type katamari.")
        resolved_checkpoint = checkpoint
    else:
        resolved_checkpoint = checkpoint or checkpoint_path(model_family, run, ckpt_name, models_root=models_root)
    predict_flags = {}
    if z_range is not None:
        predict_flags["z_range"] = list(z_range)
    if z_reduction is not None:
        predict_flags["z_reduction"] = z_reduction
    if reference_pixel_size is not None:
        predict_flags["reference_pixel_size"] = reference_pixel_size
    if reference_pixel_size_z_um is not None:
        predict_flags["reference_pixel_size_z_um"] = reference_pixel_size_z_um
    if batch_size is not None:
        predict_flags["batch_size"] = batch_size
    if yx_patch_size is not None:
        predict_flags["yx_patch_size"] = list(yx_patch_size)
    if katamari_model_input_size is not None:
        predict_flags["katamari_model_input_size"] = list(katamari_model_input_size)
    if katamari_readout is not None:
        predict_flags["katamari_readout"] = katamari_readout
    if katamari_use_teacher:
        predict_flags["katamari_use_teacher"] = True
    if z_window is not None:
        predict_flags["z_window"] = z_window
    if focus_channel is not None:
        predict_flags["focus_channel"] = focus_channel
    if z_focus_offset is not None:
        predict_flags["z_focus_offset"] = z_focus_offset

    cmd = build_predict_cmd(
        collection,
        resolved_checkpoint,
        model_family,
        run,
        ckpt_name,
        datasets_root,
        predict_flags=predict_flags or None,
        markers=list(markers) if markers else None,
        no_labelfree=no_labelfree,
        num_workers=num_workers,
        model_type=model_type,
        training_config=training_config,
        eligibility=eligibility,
    )

    if print_cmd:
        print("\n".join(cmd))
        return

    if model_type == "foundation":
        print(f"training_config: {training_config}", file=sys.stderr)
    else:
        print(f"checkpoint: {resolved_checkpoint}", file=sys.stderr)
    print(f"launching: {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
