"""Audit collection storage locations and AI-readiness from live Zarr metadata."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import click

from dynaclr.evaluation.orchestration.predict_batch import (
    SOURCE_DATASETS_ROOT,
    check_ai_ready,
)
from viscy_data.collection import load_collection

VAST_DATASETS_ROOT = Path("/hpc/projects/organelle_phenotyping/datasets")


def _storage_location(path: str | Path) -> str:
    """Classify an image store as VAST, native assemble, or another location."""
    resolved = Path(path)
    try:
        resolved.relative_to(VAST_DATASETS_ROOT)
        return "VAST"
    except ValueError:
        pass
    if any(part.endswith("-assemble") or part.endswith("-assembly") for part in resolved.parts):
        return "native assemble"
    return "other"


def build_readiness_report(collection_path: Path) -> dict:
    """Return a serializable live readiness report for one collection."""
    collection = load_collection(collection_path)
    readiness = check_ai_ready(collection_path, source_root=SOURCE_DATASETS_ROOT)
    datasets = []
    for experiment, status in zip(collection.experiments, readiness, strict=True):
        tracks_exists = Path(experiment.tracks_path).exists()
        datasets.append(
            {
                "dataset": experiment.name,
                "storage": _storage_location(experiment.data_path),
                "image_store": str(experiment.data_path),
                "tracking_store": str(experiment.tracks_path),
                "tracking_exists": tracks_exists,
                "required_channels": [channel.name for channel in experiment.channels],
                "required_markers": [channel.marker for channel in experiment.channels],
                "missing_normalization_channels": status["missing_normalization_channels"],
                "missing_focus_slice_channels": status["missing_focus_slice_channels"],
                "source_assembly": status["source_assembly"],
                "source_stale": status["stale_source"],
                "ai_ready": tracks_exists
                and not status["missing_normalization"]
                and not status["missing_focus_slice"]
                and not status["stale_source"],
            }
        )
    return {
        "collection": collection.name,
        "collection_path": str(collection_path),
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "datasets": datasets,
    }


def render_markdown(report: dict) -> str:
    """Render a concise human-readable readiness ledger."""
    lines = [
        f"# {report['collection']} AI-readiness",
        "",
        f"Generated from live Zarr metadata at `{report['checked_at']}`.",
        "Do not edit readiness fields manually; rerun `dynaclr audit-ai-readiness`.",
        "",
        "| Dataset | Storage | Required markers | Missing normalization | Missing focus | Tracks | Source current | AI-ready |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for item in report["datasets"]:
        missing_norm = ", ".join(item["missing_normalization_channels"]) or "—"
        missing_focus = ", ".join(item["missing_focus_slice_channels"]) or "—"
        lines.append(
            "| {dataset} | {storage} | {markers} | {norm} | {focus} | {tracks} | {current} | {ready} |".format(
                dataset=item["dataset"],
                storage=item["storage"],
                markers=", ".join(item["required_markers"]),
                norm=missing_norm,
                focus=missing_focus,
                tracks="yes" if item["tracking_exists"] else "no",
                current="no" if item["source_stale"] else "yes",
                ready="yes" if item["ai_ready"] else "no",
            )
        )
    lines.extend(["", "## Exact paths", ""])
    for item in report["datasets"]:
        lines.extend(
            [
                f"### {item['dataset']}",
                "",
                f"- Images: `{item['image_store']}`",
                f"- Tracks: `{item['tracking_store']}`",
                "",
            ]
        )
    return "\n".join(lines)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-c",
    "--collection",
    "collection_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Collection YAML to audit.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["markdown", "json"]),
    default="markdown",
    show_default=True,
)
@click.option("-o", "--output", type=click.Path(path_type=Path), help="Write the report to this path.")
def main(collection_path: Path, output_format: str, output: Path | None) -> None:
    """Report where collection datasets live and whether each is AI-ready."""
    report = build_readiness_report(collection_path)
    rendered = render_markdown(report) if output_format == "markdown" else json.dumps(report, indent=2) + "\n"
    if output is None:
        click.echo(rendered)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)
    click.echo(f"Wrote {output}")


if __name__ == "__main__":
    main()
