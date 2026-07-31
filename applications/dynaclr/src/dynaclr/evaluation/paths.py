"""Canonical grammar for DynaCLR embedding artifact paths.

Every per-marker embedding zarr is located by a single canonical tuple
``(dataset, model_family, run, ckpt_name, marker)``. The directory tree *is* the
index — there is no registry or manifest file. All producers and consumers
(``predict-triplet``, ``split-embeddings``, the embedding-consistency QC, MMD /
linear-classifier pooling, and the Nextflow ``eval_from_embeddings`` entry)
import these functions so the convention is defined in exactly one place.

Convention
----------
    {datasets_root}/{dataset}/2-phenotyping/predictions/
        {model_family}/{run}/{ckpt_name}/{marker}.zarr

The dataset is encoded in the *path*, so the filename is just ``{marker}.zarr``;
multiple markers of one physical dataset co-locate in the same
``{model_family}/{run}/{ckpt_name}/`` directory. "Everything computed for this
dataset" is its ``2-phenotyping/predictions/`` subtree; pooling across datasets
for one model/run/checkpoint is a single glob (:func:`iter_embeddings`).

Roots are module-level constants that every function takes as a defaulted
argument — global by default, overridable per call (pass ``datasets_root`` in
tests). The root is a fixed project fact, so it is a constant here rather than an
environment variable.
"""

from __future__ import annotations

from pathlib import Path

#: Canonical base for AI-ready datasets and their phenotyping artifacts.
DATASETS_ROOT = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics")

#: Per-dataset phenotyping subfolder (stage 2 of the dataset pipeline).
PHENOTYPING_DIR = "2-phenotyping"

#: Prediction artifacts subfolder under the phenotyping dir. Holds one
#: ``{family}/{run}/{ckpt}`` container per model, under which ``embeddings/``
#: keeps the raw per-marker zarrs alongside downstream analysis (umap/, pca/, ...).
PREDICTIONS_DIR = "predictions"

#: Subfolder (under ``{ckpt}``) holding the raw per-marker embedding zarrs,
#: kept separate from downstream analysis of those embeddings.
EMBEDDINGS_DIR = "embeddings"


def dataset_name_from_data_path(data_path: str | Path) -> str:
    """Derive the dataset name from an OME-Zarr ``data_path``.

    The dataset root is the parent of the ``{dataset}.zarr`` store, and the
    dataset name is that parent's directory name — e.g.
    ``.../organelle_dynamics/2026_07_01_A549/2026_07_01_A549.zarr`` →
    ``2026_07_01_A549``.

    Parameters
    ----------
    data_path : str or Path
        Path to the dataset OME-Zarr store (``.../{dataset}/{dataset}.zarr``).

    Returns
    -------
    str
        The dataset (directory) name.
    """
    return Path(data_path).parent.name


def dataset_root_from_data_path(data_path: str | Path) -> Path:
    """Return the dataset folder (parent of the ``{dataset}.zarr`` store).

    Parameters
    ----------
    data_path : str or Path
        Path to the dataset OME-Zarr store.

    Returns
    -------
    Path
        The dataset root directory that owns the ``2-phenotyping/`` subtree.
    """
    return Path(data_path).parent


def predictions_root(dataset: str, datasets_root: str | Path = DATASETS_ROOT) -> Path:
    """Return the ``2-phenotyping/predictions`` root for a dataset.

    Parameters
    ----------
    dataset : str
        Dataset (directory) name.
    datasets_root : str or Path, optional
        Base under which datasets live. Defaults to :data:`DATASETS_ROOT`.

    Returns
    -------
    Path
        ``{datasets_root}/{dataset}/2-phenotyping/predictions``.
    """
    return Path(datasets_root) / dataset / PHENOTYPING_DIR / PREDICTIONS_DIR


def prediction_dir(
    dataset: str,
    model_family: str,
    run: str,
    ckpt_name: str,
    datasets_root: str | Path = DATASETS_ROOT,
) -> Path:
    """Return the model/run/checkpoint-scoped directory holding per-marker zarrs.

    Parameters
    ----------
    dataset : str
        Dataset (directory) name.
    model_family, run, ckpt_name : str
        Provenance identity of the embedding-producing model.
    datasets_root : str or Path, optional
        Base under which datasets live. Defaults to :data:`DATASETS_ROOT`.

    Returns
    -------
    Path
        ``{predictions_root}/{model_family}/{run}/{ckpt_name}``.
    """
    return predictions_root(dataset, datasets_root) / model_family / run / ckpt_name


def embedding_store(
    dataset: str,
    model_family: str,
    run: str,
    ckpt_name: str,
    marker: str,
    datasets_root: str | Path = DATASETS_ROOT,
) -> Path:
    """Return the canonical per-marker embedding zarr path.

    Parameters
    ----------
    dataset : str
        Dataset (directory) name.
    model_family, run, ckpt_name : str
        Provenance identity of the embedding-producing model.
    marker : str
        Reporter/marker label; becomes the zarr filename.
    datasets_root : str or Path, optional
        Base under which datasets live. Defaults to :data:`DATASETS_ROOT`.

    Returns
    -------
    Path
        ``{prediction_dir}/embeddings/{marker}.zarr`` — the raw zarrs live in an
        ``embeddings/`` subfolder so downstream analysis (umap/, pca/, ...) can
        sit beside them under the shared ``{ckpt}`` container.
    """
    return prediction_dir(dataset, model_family, run, ckpt_name, datasets_root) / EMBEDDINGS_DIR / f"{marker}.zarr"


def iter_embeddings(
    model_family: str,
    run: str,
    ckpt_name: str,
    marker: str | None = None,
    datasets_root: str | Path = DATASETS_ROOT,
) -> list[Path]:
    """Glob per-marker embedding zarrs for one model/run/checkpoint across datasets.

    This is the pooling entry point for downstream tasks (embedding-consistency
    QC, MMD, linear classifiers): fix the model/run/checkpoint and optionally a
    marker, and collect every dataset's matching zarr.

    Parameters
    ----------
    model_family, run, ckpt_name : str
        Provenance identity to pool over.
    marker : str or None, optional
        Restrict to one marker; ``None`` matches every marker.
    datasets_root : str or Path, optional
        Base under which datasets live. Defaults to :data:`DATASETS_ROOT`.

    Returns
    -------
    list[Path]
        Sorted matching zarr paths (one per dataset x marker).
    """
    pattern = f"{marker}.zarr" if marker is not None else "*.zarr"
    glob = f"*/{PHENOTYPING_DIR}/{PREDICTIONS_DIR}/{model_family}/{run}/{ckpt_name}/{EMBEDDINGS_DIR}/{pattern}"
    return sorted(Path(datasets_root).glob(glob))
