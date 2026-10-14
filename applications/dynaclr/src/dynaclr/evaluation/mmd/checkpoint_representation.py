"""Export the canonical pooled representation for one model checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import click

from dynaclr.evaluation.mmd.export_representation import (
    DEFAULT_REPRESENTATION_KEY,
    export_pooled_representation,
    load_pooled_representation_config,
)
from dynaclr.evaluation.predict_triplet import plan_predict_runs
from viscy_data.collection import load_collection

DEFAULT_RECIPE = Path("applications/dynaclr/configs/evaluation/recipes/witness_gmm_pooled_joint_pca80.yaml")


@dataclass(frozen=True)
class CheckpointRepresentationPlan:
    """Resolved stores and shared artifact directory for one checkpoint."""

    input_paths: tuple[Path, ...]
    experiments: tuple[str, ...]
    artifact_dir: Path


def _embedding_complete(path: Path) -> bool:
    return path.is_dir() and ((path / "zarr.json").exists() or (path / ".zgroup").exists())


def plan_checkpoint_representation(
    collection_path: Path,
    *,
    model_family: str,
    run: str,
    ckpt_name: str,
    datasets_root: Path,
    markers: list[str] | None = None,
    artifact_dir: Path | None = None,
) -> CheckpointRepresentationPlan:
    """Resolve and validate every predicted store in the full collection."""
    collection = load_collection(collection_path)
    runs = plan_predict_runs(
        collection,
        model_family=model_family,
        run=run,
        ckpt_name=ckpt_name,
        datasets_root=datasets_root,
        markers=markers,
    )
    expected_experiments = {experiment.name for experiment in collection.experiments}
    planned_experiments = {item.experiment for item in runs}
    missing_experiments = sorted(expected_experiments - planned_experiments)
    if missing_experiments:
        raise ValueError(f"No requested marker stores were planned for collection experiment(s): {missing_experiments}")

    paths = tuple(dict.fromkeys(item.output_path for item in runs))
    incomplete = [str(path) for path in paths if not _embedding_complete(path)]
    if incomplete:
        preview = "\n".join(f"  - {path}" for path in incomplete[:20])
        remainder = len(incomplete) - min(len(incomplete), 20)
        suffix = f"\n  ... and {remainder} more" if remainder else ""
        raise FileNotFoundError(
            "Normalization requires every predicted store in the full collection. "
            f"Missing or incomplete ({len(incomplete)}):\n{preview}{suffix}"
        )

    artifacts = artifact_dir or (
        datasets_root / "_pooled_representation" / model_family / run / ckpt_name / DEFAULT_REPRESENTATION_KEY
    )
    return CheckpointRepresentationPlan(
        input_paths=paths,
        experiments=tuple(sorted(planned_experiments)),
        artifact_dir=artifacts,
    )


def export_checkpoint_representation(
    collection_path: Path,
    *,
    model_family: str,
    run: str,
    ckpt_name: str,
    datasets_root: Path,
    markers: list[str] | None = None,
    recipe: Path = DEFAULT_RECIPE,
    artifact_dir: Path | None = None,
    overwrite: bool = True,
):
    """Fit pooled control-MAD/PCA80 and update every checkpoint store."""
    plan = plan_checkpoint_representation(
        collection_path,
        model_family=model_family,
        run=run,
        ckpt_name=ckpt_name,
        datasets_root=datasets_root,
        markers=markers,
        artifact_dir=artifact_dir,
    )
    config = load_pooled_representation_config(recipe).model_copy(
        update={
            "input_paths": [str(path) for path in plan.input_paths],
            "output_dir": str(plan.artifact_dir.parent),
        }
    )
    return export_pooled_representation(
        config,
        artifact_dir=plan.artifact_dir,
        overwrite=overwrite,
    )


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("-c", "--collection", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--model-family", required=True)
@click.option("--run", required=True)
@click.option("--ckpt-name", required=True)
@click.option("--datasets-root", type=click.Path(path_type=Path), required=True)
@click.option("--markers", multiple=True)
@click.option("--recipe", type=click.Path(exists=True, path_type=Path), default=DEFAULT_RECIPE)
@click.option("--artifact-dir", type=click.Path(path_type=Path), default=None)
@click.option("--overwrite/--no-overwrite", default=True, show_default=True)
def main(
    collection: Path,
    model_family: str,
    run: str,
    ckpt_name: str,
    datasets_root: Path,
    markers: tuple[str, ...],
    recipe: Path,
    artifact_dir: Path | None,
    overwrite: bool,
) -> None:
    """Write pooled control-MAD/PCA80 coordinates to a checkpoint's stores."""
    manifest = export_checkpoint_representation(
        collection,
        model_family=model_family,
        run=run,
        ckpt_name=ckpt_name,
        datasets_root=datasets_root,
        markers=list(markers) or None,
        recipe=recipe,
        artifact_dir=artifact_dir,
        overwrite=overwrite,
    )
    click.echo(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
