"""Plan dataset-level phenotyping and optionally submit one Slurm job per dataset.

Register ``main`` as ``phenotyping-batch`` in the CLI. Matrix callers should
plan once across datasets, outside their model/checkpoint loops. Planning only
reads YAML and filesystem metadata; the pipeline always runs inside Slurm.

Batch YAML wraps per-dataset core configs, for example::

    datasets:
      plate_a:
        source: /source/plate_a/4-assemble/plate_a.zarr
        tracks: /data/plate_a/tracking.zarr
        focus_source: /data/plate_a/plate_a.zarr

Other top-level runner settings are preserved. A dataset entry may instead be
``{config: plate_a.yml}``, referencing a core config with ``dataset: plate_a``.
An ordinary single-dataset core config is also accepted directly.
Paths are relative to their declaring YAML. With a collection, omitted inputs
are derived from the canonical source assembly, collection tracks, and the
collection image store carrying focus metadata. Each materialized core config
has top-level ``dataset/source/focus_source/tracks/output`` fields. Outputs
allocate ``00_cpdino_baseline``, ``01_cpdino_baseline``, ... directly under
``DATASETS_ROOT/DATASET/2-phenotyping/segmentations_membrane``. An existing
attempt is never reused. A core ``output`` ending in ``NN_cpdino_baseline``
selects its parent as the attempt root; other output paths are attempt roots.

No arguments launch work by default. ``--submit`` requires one or more explicit
``--experiment NAME`` selections. ``--dry-run`` and ``--submit`` are exclusive.
Use ``.venv-phenotyping/bin/python -m`` to invoke this module, or pass
``--python /absolute/path/to/.venv-phenotyping/bin/python`` from another CLI
environment. Matrix callers pass the same path as ``runner_python``.
"""

from __future__ import annotations

import copy
import json
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import click
import yaml

from dynaclr.evaluation.orchestration.predict_batch import SOURCE_DATASETS_ROOT, _latest_source_assembly
from dynaclr.evaluation.paths import DATASETS_ROOT
from viscy_data.collection import load_collection

_INPUTS = ("source", "tracks", "focus_source")
_BASELINE = Path("2-phenotyping/segmentations_membrane")


@dataclass(frozen=True)
class PhenotypingRun:
    """One dataset's resolved runner configuration and output location."""

    dataset: str
    config: dict
    output: Path
    output_base: Path
    config_path: Path
    runner_python: Path | None = None

    @property
    def key(self) -> tuple[Path, str]:
        """Identity shared across model rows and overlapping collections."""
        return self.config_path, self.dataset


def _read_config(path: Path) -> dict:
    with path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    if any(key in config for key in _INPUTS) and not config.get("dataset"):
        raise ValueError(f"{path}: input stores require an explicit 'dataset' or per-dataset 'datasets' mapping")
    if "dataset" in config and "datasets" in config:
        raise ValueError(f"{path}: use either a single 'dataset' or a batch 'datasets' mapping")
    datasets = config.get("datasets", {})
    if not isinstance(datasets, dict):
        raise ValueError(f"{path}: 'datasets' must be a mapping keyed by dataset name")
    return config


def _path(value: str | Path, base: Path) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"Expected a nonempty path, got {value!r}")
    path = Path(value).expanduser()
    return (base / path).resolve()


def _resolve_recipe_paths(config: dict, base: Path) -> None:
    """Keep inherited file references anchored to the YAML that declared them."""
    for annotation in config.get("annotations", []):
        annotation["path"] = str(_path(annotation["path"], base))
    segmentation = config.get("segmentation", {})
    if segmentation.get("checkpoint", "cpdino") != "cpdino":
        segmentation["checkpoint"] = str(_path(segmentation["checkpoint"], base))
    if "output" in config:
        config["output"] = str(_path(config["output"], base))


def _next_attempt(base: Path) -> Path:
    numbers = [
        int(match.group(1))
        for path in (base.iterdir() if base.is_dir() else ())
        if (match := re.match(r"(\d+)_", path.name))
    ]
    return base / f"{max(numbers, default=-1) + 1:02d}_cpdino_baseline"


def _allocate_attempt(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    while True:
        attempt = _next_attempt(base)
        try:
            attempt.mkdir()
            return attempt
        except FileExistsError:
            # Another submitter reserved this attempt after our directory scan.
            continue


def _metadata(path: Path) -> dict:
    """Read Zarr attributes only, without importing array or GPU libraries."""
    if (path / ".zattrs").is_file():
        with (path / ".zattrs").open() as stream:
            return json.load(stream)
    if (path / "zarr.json").is_file():
        with (path / "zarr.json").open() as stream:
            return json.load(stream).get("attributes", {})
    return {}


def _is_volume(path: Path, channels: dict) -> bool:
    plate = _metadata(path)
    wells = plate.get("plate", {}).get("wells", [])
    if not wells:
        return False
    well = path / wells[0]["path"]
    images = _metadata(well).get("well", {}).get("images", [])
    if not images:
        return False
    position = well / images[0]["path"]
    attrs = _metadata(position)
    labels = {channel.get("label") for channel in attrs.get("omero", {}).get("channels", [])}
    expected = {channels.get("membrane", "membrane_prediction"), channels.get("nuclei", "nuclei_prediction")}
    if not expected <= labels or not attrs.get("multiscales"):
        return False
    multiscale = attrs["multiscales"][0]
    axes = [axis["name"] if isinstance(axis, dict) else axis for axis in multiscale.get("axes", [])]
    if "z" not in axes:
        return False
    array = position / multiscale["datasets"][0]["path"]
    metadata = array / ".zarray"
    if not metadata.is_file():
        metadata = array / "zarr.json"
    if not metadata.is_file():
        return False
    with metadata.open() as stream:
        shape = json.load(stream)["shape"]
    return shape[axes.index("z")] >= 5


def _resolve_source_assembly(name: str, source_root: Path, channels: dict) -> Path:
    # Reuse prediction discovery, but never let its mtime ordering choose a MIP
    # or resolve a scientific ambiguity between different assembled volumes.
    latest = _latest_source_assembly(name, source_root)
    candidates = sorted(
        {
            path
            for pattern in ("*-assemble/*.zarr", "*-assembly/*.zarr")
            for path in (source_root / name).glob(pattern)
            if path.is_dir() and not path.stem.lower().endswith("_mip")
        }
    )
    exact = [path for path in candidates if path.name == f"{name}.zarr"]
    if len(exact) == 1:
        return exact[0]
    if not exact:
        if latest in candidates:
            candidates.remove(latest)
            candidates.insert(0, latest)
        candidates = [path for path in candidates if _is_volume(path, channels)]
    else:
        candidates = exact
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        raise ValueError(f"{name}: ambiguous source assemblies; provide an explicit source: {candidates}")
    raise ValueError(f"{name}: no canonical source assembly; provide an explicit source")


def plan_phenotyping_runs(
    config: Path,
    *,
    collection: Path | None = None,
    experiments: tuple[str, ...] = (),
    source_root: Path = SOURCE_DATASETS_ROOT,
    output_root: Path | None = None,
    datasets_root: Path = DATASETS_ROOT,
) -> list[PhenotypingRun]:
    """Resolve and validate every selected dataset before any submission.

    Input stores must exist, but array contents and focus metadata validation
    belong to the compute-side runner. Returned configs use the core's flat
    schema. Explicit dataset mappings take precedence over collection defaults.
    Duplicate selections are collapsed in first-seen order.
    """
    config = config.resolve()
    pilot = _read_config(config)
    if "dataset" in pilot:
        entries = {pilot["dataset"]: pilot}
        shared = {}
    else:
        entries = pilot.get("datasets", {})
        shared = {key: value for key, value in pilot.items() if key != "datasets"}
    collection_entries = {}
    if collection is not None:
        collection = collection.resolve()
        collection_entries = {entry.name: entry for entry in load_collection(collection).experiments}
        unknown = entries.keys() - collection_entries.keys()
        if unknown:
            raise ValueError(f"Config datasets absent from collection: {', '.join(sorted(unknown))}")
    # A named core pilot never silently fans out to unrelated collection datasets.
    names = list(entries) if "dataset" in pilot or collection is None else list(collection_entries)
    if experiments:
        unknown = set(experiments) - set(names)
        if unknown:
            raise ValueError(f"Unknown experiment(s): {', '.join(sorted(unknown))}")
        names = list(dict.fromkeys(experiments))
    if not names:
        raise ValueError("No datasets: provide a collection or explicit per-dataset configuration")

    runs = []
    for name in names:
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError(f"Invalid dataset name: {name!r}")
        entry = entries.get(name, {})
        if not isinstance(entry, dict):
            raise ValueError(f"{name}: dataset configuration must be a mapping")
        base = config.parent
        resolved = copy.deepcopy(shared)
        _resolve_recipe_paths(resolved, base)
        config_path = config
        if "config" in entry:
            if set(entry) != {"config"}:
                raise ValueError(f"{name}: 'config' references cannot also contain overrides")
            dataset_config = _path(entry["config"], base)
            specific = _read_config(dataset_config)
            if specific.get("dataset") != name:
                raise ValueError(f"{dataset_config}: expected dataset {name!r}")
            entry = specific
            config_path = dataset_config
            base = dataset_config.parent
        entry = copy.deepcopy(entry)
        _resolve_recipe_paths(entry, base)
        if entry.get("dataset", name) != name:
            raise ValueError(f"{name}: dataset name does not match its configuration")
        resolved.update(entry)
        entry = resolved
        entry["dataset"] = name
        if name in collection_entries:
            experiment = collection_entries[name]
            if "source" not in entry:
                assembly = _resolve_source_assembly(name, source_root.resolve(), entry.get("channels", {}))
                entry["source"] = str(assembly)
            # Collection paths are relative to the collection, not the pilot.
            if "tracks" not in entry:
                entry["tracks"] = str(_path(experiment.tracks_path, collection.parent))
            if "focus_source" not in entry:
                entry["focus_source"] = str(_path(experiment.data_path, collection.parent))
            entry["exclude_fovs"] = list(
                dict.fromkeys(
                    [
                        *entry.get("exclude_fovs", []),
                        *experiment.exclude_fovs,
                    ]
                )
            )
        for key in _INPUTS:
            if key not in entry:
                raise ValueError(f"{name}: missing required per-dataset {key}")
            path = _path(entry[key], base)
            if not path.is_dir():
                raise ValueError(f"{name}: {key} is not an existing store/directory: {path}")
            entry[key] = str(path)
        channels = entry.get("channels")
        if not isinstance(channels, dict) or not channels.get("fluorescence"):
            raise ValueError(f"{name}: explicit channels.fluorescence is required (name, marker, role per channel)")
        fluorescence = channels["fluorescence"]
        if not isinstance(fluorescence, list) or any(
            not isinstance(channel, dict) or not channel.get("name") or not channel.get("marker")
            for channel in fluorescence
        ):
            raise ValueError(f"{name}: channels.fluorescence must list channel name/marker mappings")
        if name in collection_entries:
            for channel in collection_entries[name].channels:
                configured = [item for item in fluorescence if item["name"] == channel.name]
                matches = [item for item in configured if item["marker"] == channel.marker]
                if configured and not matches:
                    raise ValueError(f"{name}: channels.fluorescence is missing per-well marker {channel.marker}")
                for item in matches:
                    if channel.wells:
                        if item.get("wells") and not set(item["wells"]) <= set(channel.wells):
                            raise ValueError(f"{name}: {channel.marker} wells disagree with the collection")
                        item.setdefault("wells", list(channel.wells))
                        if not item["wells"]:
                            item["wells"] = list(channel.wells)
        for index, channel in enumerate(fluorescence):
            for other in fluorescence[:index]:
                if channel["name"] == other["name"] or channel["marker"] == other["marker"]:
                    wells, previous = set(channel.get("wells", [])), set(other.get("wells", []))
                    if not wells or not previous or wells & previous:
                        raise ValueError(f"{name}: repeated fluorescence names/markers require disjoint wells")
        if output_root is not None:
            output_base = output_root.resolve() / name
        elif "output" in entry:
            output_base = _path(entry["output"], base)
            if re.fullmatch(r"\d+_cpdino_baseline", output_base.name):
                output_base = output_base.parent
        else:
            output_base = datasets_root.resolve() / name / _BASELINE
        output = _next_attempt(output_base)
        entry["output"] = str(output)
        runs.append(PhenotypingRun(name, entry, output, output_base, config_path))
    outputs = [run.output for run in runs]
    if len(set(outputs)) != len(outputs):
        raise ValueError("Datasets must have distinct output paths")
    return runs


def plan_matrix_phenotyping(
    models: list[dict],
    *,
    stages: tuple[str, ...] | list[str],
    experiments: tuple[str, ...] = (),
    source_root: Path = SOURCE_DATASETS_ROOT,
) -> list[PhenotypingRun]:
    """Plan optional ``phenotype`` work once before the matrix's model loop.

    Resolved rows inherit ``defaults.phenotyping_config`` and optionally
    ``defaults.phenotyping_python`` (set to the isolated runner interpreter).
    Without an explicit
    ``phenotype`` stage this returns an empty plan without reading any files.
    Submit the returned plan once; the returned job IDs from
    :func:`submit_phenotyping_runs` can be shared by all dependent model jobs.
    """
    if "phenotype" not in stages:
        return []
    runs = {}
    datasets = {}
    seen = set()
    for model in models:
        if not model.get("phenotyping_config"):
            continue
        config = Path(model["phenotyping_config"]).resolve()
        collection = Path(model["collection"]).resolve() if model.get("collection") else None
        identity = (config, collection, str(model.get("datasets_root", DATASETS_ROOT)), model.get("phenotyping_python"))
        if identity in seen:
            continue
        seen.add(identity)
        for run in plan_phenotyping_runs(
            config,
            collection=collection,
            experiments=experiments,
            source_root=source_root,
            datasets_root=Path(model.get("datasets_root", DATASETS_ROOT)),
        ):
            if model.get("phenotyping_python"):
                run = replace(run, runner_python=Path(model["phenotyping_python"]).absolute())
            if run.dataset in datasets and datasets[run.dataset] != run:
                raise ValueError(f"Conflicting phenotyping configurations for dataset {run.dataset}")
            if run.key in runs and runs[run.key] != run:
                raise ValueError(f"Conflicting phenotyping mappings for {run.dataset} in {run.config_path}")
            datasets[run.dataset] = run
            runs[run.key] = run
    if not runs:
        raise ValueError("Stage 'phenotype' requires defaults.phenotyping_config or a per-row phenotyping_config")
    return list(runs.values())


def build_phenotyping_cmd(
    config: Path,
    *,
    dataset: str | None = None,
    output: Path | None = None,
    dry_run: bool = False,
    runner_python: Path | str | None = None,
) -> list[str]:
    """Build the core runner interface without invoking it."""
    command = [str(runner_python or sys.executable), "-m", "dynaclr.phenotyping.pipeline", "--config", str(config)]
    if dataset is not None:
        command += ["--dataset", dataset]
    if output is not None:
        command += ["--output", str(output)]
    if dry_run:
        command.append("--dry-run")
    return command


def build_sbatch_cmd(
    run: PhenotypingRun,
    config: Path,
    *,
    workspace: Path,
    partition: str = "gpu",
    cpus: int = 4,
    gpus: int = 1,
    memory: str = "64G",
    time_limit: str = "72:00:00",
    runner_python: Path | str | None = None,
) -> list[str]:
    """Wrap one dataset runner in Slurm, quoting every runner argument."""
    command = [
        "sbatch",
        "--parsable",
        f"--job-name=phenotyping-{run.dataset}",
        f"--chdir={workspace.resolve()}",
        "--nodes=1",
        "--ntasks-per-node=1",
        f"--cpus-per-task={cpus}",
        f"--mem={memory}",
        f"--time={time_limit}",
        f"--output={config.parent / (run.dataset + '-%j.out')}",
    ]
    if partition is not None:
        command.append(f"--partition={partition}")
    if gpus:
        command.append(f"--gpus={gpus}")
    command += [
        "--wrap",
        shlex.join(
            [
                "env",
                f"OPENBLAS_NUM_THREADS={cpus}",
                f"OMP_NUM_THREADS={cpus}",
                *build_phenotyping_cmd(
                    config,
                    dataset=run.dataset,
                    output=run.output,
                    runner_python=runner_python or run.runner_python,
                ),
            ]
        ),
    ]
    return command


def submit_phenotyping_runs(
    runs: list[PhenotypingRun],
    *,
    config_dir: Path,
    workspace: Path,
    partition: str = "gpu",
    cpus: int = 4,
    gpus: int = 1,
    memory: str = "64G",
    time_limit: str = "72:00:00",
    runner_python: Path | str | None = None,
) -> dict[tuple[Path, str], str]:
    """Persist unique job configs, submit once per dataset, and return job IDs.

    Call once with the complete plan, outside model loops. Job IDs are printed
    immediately so an earlier successful submission remains visible if a later
    submission fails. This function never retries submissions automatically.
    """
    if not runs or len({run.key for run in runs}) != len(runs):
        raise ValueError("Submission requires a nonempty plan with unique config/dataset pairs")
    if len({run.dataset for run in runs}) != len(runs):
        raise ValueError("Conflicting phenotyping configurations for the same dataset")
    config_dir = config_dir.resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    batch_dir = Path(tempfile.mkdtemp(prefix="batch-", dir=config_dir))
    commands = []
    for index, run in enumerate(runs):
        output = _allocate_attempt(run.output_base)
        resolved = copy.deepcopy(run.config)
        resolved["output"] = str(output)
        run = replace(run, config=resolved, output=output)
        config = batch_dir / f"{index:04d}-{run.dataset}.yml"
        with config.open("x") as stream:
            yaml.safe_dump(run.config, stream, sort_keys=False)
        commands.append(
            build_sbatch_cmd(
                run,
                config,
                workspace=workspace,
                partition=partition,
                cpus=cpus,
                gpus=gpus,
                memory=memory,
                time_limit=time_limit,
                runner_python=runner_python,
            )
        )
    jobs = {}
    for run, command in zip(runs, commands, strict=True):
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        job_id = result.stdout.strip().split(";", 1)[0]
        if not job_id.isdecimal():
            raise ValueError(
                f"{run.dataset}: unexpected sbatch response {result.stdout!r}; check Slurm before retrying"
            )
        jobs[run.key] = job_id
        click.echo(f"{run.dataset}: submitted job {job_id}")
    return jobs


def print_phenotyping_plan(runs: list[PhenotypingRun], *, config_dir: Path, workspace: Path, **resources) -> None:
    """Print resolved configs and prospective submissions without filesystem writes."""
    click.echo("Dry run: no files written or jobs submitted. Config paths below are preview locations.")
    for index, run in enumerate(runs):
        config_path = config_dir.resolve() / "preview" / f"{index:04d}-{run.dataset}.yml"
        click.echo(f"\n# {run.dataset}: {config_path}")
        click.echo(yaml.safe_dump(run.config, sort_keys=False).rstrip())
        click.echo(shlex.join(build_sbatch_cmd(run, config_path, workspace=workspace, **resources)))


@click.command("phenotyping-batch", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("-c", "--collection", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--config", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--experiment", "experiments", multiple=True, help="Dataset name; repeat to select several. Required to submit."
)
@click.option("--dry-run", is_flag=True, help="Print the resolved plan without writes or launches (the default).")
@click.option("--submit", is_flag=True, help="Submit one Slurm job per selected dataset.")
@click.option("--source-root", type=click.Path(path_type=Path), default=SOURCE_DATASETS_ROOT, show_default=True)
@click.option("--datasets-root", type=click.Path(path_type=Path), default=DATASETS_ROOT, show_default=True)
@click.option("--output-root", type=click.Path(path_type=Path), help="Outputs go under ROOT/DATASET.")
@click.option("--config-dir", type=click.Path(path_type=Path), default=Path(".phenotyping-batch"), show_default=True)
@click.option(
    "--workspace-dir", "workspace", type=click.Path(exists=True, file_okay=False, path_type=Path), default="."
)
@click.option("--partition", default="gpu", show_default=True, help="Slurm partition.")
@click.option(
    "--python",
    "runner_python",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Runner interpreter; default is this CLI's Python. Use .venv-phenotyping/bin/python.",
)
@click.option("--cpus", type=click.IntRange(min=1), default=4, show_default=True)
@click.option("--gpus", type=click.IntRange(min=0), default=1, show_default=True)
@click.option("--memory", default="64G", show_default=True)
@click.option("--time-limit", default="72:00:00", show_default=True)
def main(
    collection: Path | None,
    config: Path,
    experiments: tuple[str, ...],
    dry_run: bool,
    submit: bool,
    source_root: Path,
    datasets_root: Path,
    output_root: Path | None,
    config_dir: Path,
    workspace: Path,
    partition: str,
    cpus: int,
    gpus: int,
    memory: str,
    time_limit: str,
    runner_python: Path | None,
) -> None:
    """Plan phenotyping; launch only with --submit and explicit --experiment selections."""
    if dry_run and submit:
        raise click.UsageError("--dry-run and --submit are mutually exclusive")
    if submit and not experiments:
        raise click.UsageError("--submit requires explicit --experiment NAME (repeat for multiple datasets)")
    try:
        runs = plan_phenotyping_runs(
            config,
            collection=collection,
            experiments=experiments,
            source_root=source_root,
            output_root=output_root,
            datasets_root=datasets_root,
        )
        resources = dict(
            workspace=workspace,
            partition=partition,
            cpus=cpus,
            gpus=gpus,
            memory=memory,
            time_limit=time_limit,
            runner_python=runner_python.absolute() if runner_python else None,
        )
        if submit:
            submit_phenotyping_runs(runs, config_dir=config_dir, **resources)
            return
        print_phenotyping_plan(runs, config_dir=config_dir, **resources)
    except (ValueError, OSError, yaml.YAMLError) as error:
        raise click.ClickException(str(error)) from error
    except subprocess.CalledProcessError as error:
        raise click.ClickException(f"Slurm submission failed: {error.stderr or error}") from error


if __name__ == "__main__":
    main()
