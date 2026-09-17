r"""Run evaluation over pre-computed embeddings for a model/run/checkpoint.

The DynaCLR analog of dynacell's ``submit_evaluation_batch.py``: turn a
``(model_family, run, ckpt_name)`` identity into the ``--embeddings_glob`` that
selects a cohort of already-written embedding zarrs, and launch the Nextflow
``eval_from_embeddings`` entry over them. No GPU / no re-prediction — evaluation
reads the frozen embeddings written by ``predict-triplet`` (or the parquet spine)
under the canonical tree
``<dataset>/2-phenotyping/predictions/{model_family}/{run}/{ckpt_name}/{marker}.zarr``.

Dataset selection is the glob: with no ``--datasets`` every dataset present in
the tree for the given identity is included (the progressive default); pass
``--datasets`` to pin a reproducible subset (brace-expanded into the glob).

Output:
  * ``--print-cmd``: print the ``nextflow run`` command (one token per line).
  * default: execute it via ``subprocess.run``.

Usage::

    dynaclr eval \
        --eval-config /path/to/eval.yaml \
        --model-family DynaCLR-2D-MIP-BagOfChannels \
        --run 2d-mip-fix-shuffler \
        --ckpt-name epoch105-step84800 \
        [--marker SEC61B] [--datasets ds_a ds_b] \
        [--print-cmd]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click

from dynaclr.evaluation.paths import DATASETS_ROOT, EMBEDDINGS_DIR, PHENOTYPING_DIR, PREDICTIONS_DIR

# Repo-relative path to the Nextflow router.
_MAIN_NF = Path("applications/dynaclr/nextflow/main.nf")


def build_embeddings_glob(
    model_family: str,
    run: str,
    ckpt_name: str,
    marker: str | None = None,
    datasets: list[str] | None = None,
    datasets_root: str | Path = DATASETS_ROOT,
) -> str:
    """Build the ``--embeddings_glob`` selecting a cohort of embedding zarrs.

    Parameters
    ----------
    model_family, run, ckpt_name : str
        Provenance identity pinning which embeddings to evaluate.
    marker : str or None, optional
        Restrict to one marker (``{marker}.zarr``); ``None`` matches all
        markers (``*.zarr``).
    datasets : list[str] or None, optional
        Explicit dataset subset, brace-expanded into the dataset slot. ``None``
        globs every dataset (``*``) present for this identity.
    datasets_root : str or Path, optional
        Base under which datasets live. Defaults to
        :data:`dynaclr.evaluation.paths.DATASETS_ROOT`.

    Returns
    -------
    str
        A shell glob string (brace expansion resolved by the shell / Nextflow).
    """
    if datasets:
        ds_slot = "{" + ",".join(datasets) + "}" if len(datasets) > 1 else datasets[0]
    else:
        ds_slot = "*"
    marker_slot = f"{marker}.zarr" if marker else "*.zarr"
    return str(
        Path(datasets_root)
        / ds_slot
        / PHENOTYPING_DIR
        / PREDICTIONS_DIR
        / model_family
        / run
        / ckpt_name
        / EMBEDDINGS_DIR
        / marker_slot
    )


def build_nextflow_cmd(
    eval_config: Path,
    embeddings_glob: str,
    workspace_dir: str,
    resume: bool,
) -> list[str]:
    """Build the ``nextflow run ... -entry eval_from_embeddings`` command."""
    cmd = [
        "nextflow",
        "run",
        str(_MAIN_NF),
        "-entry",
        "eval_from_embeddings",
        "--eval_config",
        str(eval_config),
        "--embeddings_glob",
        embeddings_glob,
        "--workspace_dir",
        workspace_dir,
    ]
    if resume:
        cmd.append("-resume")
    return cmd


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--eval-config",
    type=click.Path(path_type=Path),
    required=True,
    help="Evaluation config YAML (what steps to run).",
)
@click.option("--model-family", required=True, help="Model family (embedding-tree key).")
@click.option("--run", required=True, help="Training run/version (embedding-tree key).")
@click.option("--ckpt-name", required=True, help="Checkpoint label (embedding-tree key).")
@click.option("--marker", default=None, help="Restrict to one marker (default: all markers).")
@click.option(
    "--datasets",
    multiple=True,
    default=None,
    help="Explicit dataset subset (default: all datasets present for this model/run/ckpt).",
)
@click.option(
    "--datasets-root",
    type=click.Path(path_type=Path),
    default=DATASETS_ROOT,
    show_default=True,
    help="Base under which datasets live.",
)
@click.option(
    "--workspace-dir",
    default="/hpc/mydata/eduardo.hirata/repos/viscy",
    help="uv workspace dir passed to the Nextflow run.",
)
@click.option("--no-resume", is_flag=True, help="Do not pass -resume to Nextflow.")
@click.option("--print-cmd", is_flag=True, help="Print the command instead of executing it.")
def main(
    eval_config: Path,
    model_family: str,
    run: str,
    ckpt_name: str,
    marker: str | None,
    datasets: tuple[str, ...],
    datasets_root: Path,
    workspace_dir: str,
    no_resume: bool,
    print_cmd: bool,
) -> None:
    """Launch eval_from_embeddings over a model/run/ckpt's embeddings."""
    embeddings_glob = build_embeddings_glob(
        model_family,
        run,
        ckpt_name,
        marker=marker,
        datasets=list(datasets) if datasets else None,
        datasets_root=datasets_root,
    )
    cmd = build_nextflow_cmd(
        eval_config,
        embeddings_glob,
        workspace_dir,
        resume=not no_resume,
    )

    if print_cmd:
        print("\n".join(cmd))
        return

    print(f"embeddings_glob: {embeddings_glob}", file=sys.stderr)
    print(f"launching: {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
