"""Dataset planning and Slurm boundary tests; no phenotyping compute is launched."""

import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from click.testing import CliRunner

from dynaclr.evaluation.orchestration import phenotyping

CHANNELS = {
    "membrane": "membrane_prediction",
    "nuclei": "nuclei_prediction",
    "focus": "Phase3D",
    "fluorescence": [{"name": "GFP", "marker": "SEC61B", "role": "organelle"}],
}


def write_yaml(path, data):
    if isinstance(data, dict):
        for experiment in data.get("experiments", []):
            experiment.setdefault("perturbation_wells", {"uninfected": ["A/1", "B/1"]})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.fixture
def pilot(tmp_path):
    datasets = {}
    for name in ("plate_a", "plate_b"):
        entry = {}
        for key in ("source", "tracks", "focus_source"):
            store = tmp_path / name / f"{key}.zarr"
            store.mkdir(parents=True)
            entry[key] = str(store.relative_to(tmp_path))
        datasets[name] = entry
    return write_yaml(tmp_path / "pilot.yml", {"channels": CHANNELS, "datasets": datasets})


@pytest.fixture
def forbid_launch(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Planning must never launch a subprocess")

    monkeypatch.setattr(phenotyping.subprocess, "run", fail)


@pytest.mark.parametrize("flags", [[], ["--dry-run"]])
def test_default_and_explicit_dry_run_are_read_only(pilot, tmp_path, forbid_launch, flags):
    before = set(tmp_path.rglob("*"))
    original = pilot.read_bytes()
    result = CliRunner().invoke(
        phenotyping.main,
        [
            "--config",
            str(pilot),
            "--config-dir",
            str(tmp_path / "jobs"),
            "--datasets-root",
            str(tmp_path / "outputs"),
            *flags,
        ],
    )
    assert result.exit_code == 0, result.output
    assert "no files written or jobs submitted" in result.output
    assert result.output.count("--job-name=phenotyping-") == 2
    assert "dynaclr.phenotyping.pipeline" in result.output
    assert "source:" in result.output
    assert set(tmp_path.rglob("*")) == before
    assert pilot.read_bytes() == original


def test_explicit_mapping_is_absolute_and_selected_once(pilot, tmp_path):
    runs = phenotyping.plan_phenotyping_runs(
        pilot, experiments=("plate_b", "plate_b"), output_root=tmp_path / "results"
    )
    assert [run.dataset for run in runs] == ["plate_b"]
    run = runs[0]
    assert run.output == tmp_path / "results" / "plate_b" / "00_cpdino_baseline"
    assert run.config["dataset"] == "plate_b"
    assert "datasets" not in run.config
    assert run.config["channels"] == CHANNELS
    assert run.config["tracks"] == str(tmp_path / "plate_b" / "tracks.zarr")


def test_collection_resolves_latest_canonical_assembly_and_focus(pilot, tmp_path, forbid_launch):
    source = tmp_path / "source"
    old = source / "plate_a" / "2-assemble" / "old.zarr"
    new = source / "plate_a" / "4-assembly" / "plate_a.zarr"
    archive = source / "plate_a" / "4-assembly" / "archive" / "ignored.zarr"
    for index, store in enumerate((old, new, archive), start=1):
        store.mkdir(parents=True)
        metadata = store / ".zattrs"
        metadata.write_text("{}")
        os.utime(metadata, (index, index))
    collection = write_yaml(
        tmp_path / "collection.yml",
        {
            "name": "collection",
            "datasets_root": str(tmp_path),
            "experiments": [
                {
                    "name": "plate_a",
                    "data_path": "${datasets_root}/plate_a/focus_source.zarr",
                    "tracks_path": "${datasets_root}/plate_a/tracks.zarr",
                }
            ],
        },
    )
    config = write_yaml(tmp_path / "generic.yml", {"channels": CHANNELS})
    (run,) = phenotyping.plan_phenotyping_runs(
        config, collection=collection, source_root=source, datasets_root=tmp_path / "outputs"
    )
    assert run.config == {
        "source": str(new),
        "tracks": str(tmp_path / "plate_a" / "tracks.zarr"),
        "focus_source": str(tmp_path / "plate_a" / "focus_source.zarr"),
        "output": str(
            tmp_path / "outputs" / "plate_a" / "2-phenotyping" / "segmentations_membrane" / "00_cpdino_baseline"
        ),
        "dataset": "plate_a",
        "exclude_fovs": [],
        "channels": CHANNELS,
    }

    result = CliRunner().invoke(
        phenotyping.main,
        ["-c", str(collection), "--config", str(config), "--source-root", str(source), "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert str(new) in result.output


def test_collection_cannot_fall_back_to_ai_ready_as_assembly(pilot, tmp_path):
    collection = write_yaml(
        tmp_path / "collection.yml",
        {
            "name": "collection",
            "experiments": [
                {
                    "name": "plate_a",
                    "data_path": str(tmp_path / "plate_a" / "focus_source.zarr"),
                    "tracks_path": str(tmp_path / "plate_a" / "tracks.zarr"),
                }
            ],
        },
    )
    config = write_yaml(tmp_path / "generic.yml", {})
    with pytest.raises(ValueError, match="no canonical source assembly"):
        phenotyping.plan_phenotyping_runs(config, collection=collection, source_root=tmp_path / "missing")
    # Explicit input mapping is sufficient even without a discoverable source.
    runs = phenotyping.plan_phenotyping_runs(
        write_yaml(
            tmp_path / "explicit.yml",
            {"channels": CHANNELS, "datasets": {"plate_a": yaml.safe_load(pilot.read_text())["datasets"]["plate_a"]}},
        ),
        collection=collection,
        source_root=tmp_path / "missing",
    )
    assert len(runs) == 1


def test_per_dataset_config_reference_resolves_from_its_directory(pilot, tmp_path):
    entry = yaml.safe_load(pilot.read_text())["datasets"]["plate_a"]
    entry = {key: "../" + value for key, value in entry.items()}
    write_yaml(tmp_path / "configs" / "a.yml", {"dataset": "plate_a", **entry, "channels": CHANNELS, "max_overlays": 2})
    batch = write_yaml(tmp_path / "batch.yml", {"datasets": {"plate_a": {"config": "configs/a.yml"}}})
    (run,) = phenotyping.plan_phenotyping_runs(batch)
    assert run.config["source"] == str(tmp_path / "plate_a" / "source.zarr")
    assert run.config["max_overlays"] == 2


def test_referenced_config_preserves_shared_relative_paths(pilot, tmp_path):
    entry = yaml.safe_load(pilot.read_text())["datasets"]["plate_a"]
    specific = {"dataset": "plate_a", **{key: "../" + value for key, value in entry.items()}}
    write_yaml(tmp_path / "configs" / "a.yml", specific)
    batch = write_yaml(
        tmp_path / "batch.yml",
        {
            "channels": CHANNELS,
            "datasets": {"plate_a": {"config": "configs/a.yml"}},
            "annotations": [{"path": "events.csv", "dataset": "plate_a"}],
            "segmentation": {"checkpoint": "weights.pt"},
            "output": "results/00_cpdino_baseline",
        },
    )
    (run,) = phenotyping.plan_phenotyping_runs(batch)
    assert run.config["annotations"][0]["path"] == str(tmp_path / "events.csv")
    assert run.config["segmentation"]["checkpoint"] == str(tmp_path / "weights.pt")
    assert run.output_base == tmp_path / "results"


@pytest.mark.parametrize("key", ["source", "tracks", "focus_source"])
def test_missing_mapping_rejected_before_any_submission(pilot, tmp_path, key, forbid_launch):
    config = yaml.safe_load(pilot.read_text())
    del config["datasets"]["plate_b"][key]
    write_yaml(pilot, config)
    result = CliRunner().invoke(
        phenotyping.main,
        [
            "--config",
            str(pilot),
            "--submit",
            "--experiment",
            "plate_a",
            "--experiment",
            "plate_b",
            "--config-dir",
            str(tmp_path / "jobs"),
            "--datasets-root",
            str(tmp_path / "outputs"),
        ],
    )
    assert result.exit_code != 0
    assert f"plate_b: missing required per-dataset {key}" in result.output
    assert not (tmp_path / "jobs").exists()


@pytest.mark.parametrize(
    "flags, message",
    [
        (["--submit"], "requires explicit --experiment"),
        (["--submit", "--dry-run", "--experiment", "plate_a"], "mutually exclusive"),
        (["--experiment", "typo"], "Unknown experiment"),
    ],
)
def test_invalid_launch_selection(pilot, flags, message, forbid_launch):
    result = CliRunner().invoke(phenotyping.main, ["--config", str(pilot), *flags])
    assert result.exit_code != 0
    assert message in result.output


@pytest.mark.parametrize(
    "config, message",
    [
        ([], "expected a YAML mapping"),
        ({}, "No datasets"),
        ({"datasets": []}, "must be a mapping"),
        ({"source": "/shared"}, "require an explicit"),
        ({"datasets": {"../escape": {}}}, "Invalid dataset name"),
        ({"datasets": {"plate_a": None}}, "must be a mapping"),
    ],
)
def test_invalid_config(tmp_path, config, message):
    with pytest.raises(ValueError, match=message):
        phenotyping.plan_phenotyping_runs(write_yaml(tmp_path / "bad.yml", config))


def test_missing_store_and_output_collision(pilot, tmp_path):
    config = yaml.safe_load(pilot.read_text())
    config["datasets"]["plate_a"]["tracks"] = "missing.zarr"
    write_yaml(pilot, config)
    with pytest.raises(ValueError, match="not an existing store"):
        phenotyping.plan_phenotyping_runs(pilot)
    config["datasets"]["plate_a"]["tracks"] = "plate_a/tracks.zarr"
    for entry in config["datasets"].values():
        entry["output"] = "same-output"
    write_yaml(pilot, config)
    with pytest.raises(ValueError, match="distinct output paths"):
        phenotyping.plan_phenotyping_runs(pilot)


def test_runner_interface_and_shell_quoting(pilot, tmp_path):
    config = tmp_path / "config with 'quotes' $(whoami).yml"
    output = tmp_path / "output with spaces"
    command = phenotyping.build_phenotyping_cmd(config, dataset="plate_a", output=output, dry_run=True)
    assert command == [
        sys.executable,
        "-m",
        "dynaclr.phenotyping.pipeline",
        "--config",
        str(config),
        "--dataset",
        "plate_a",
        "--output",
        str(output),
        "--dry-run",
    ]
    assert phenotyping.build_phenotyping_cmd(config) == command[:5]
    run = phenotyping.plan_phenotyping_runs(pilot)[0]
    sbatch = phenotyping.build_sbatch_cmd(run, config, workspace=tmp_path, partition="cpu", cpus=2)
    assert "--partition=cpu" in sbatch
    assert "--cpus-per-task=2" in sbatch
    assert shlex.split(sbatch[-1])[3:] == phenotyping.build_phenotyping_cmd(
        config, dataset="plate_a", output=run.output
    )


def test_submit_once_per_dataset_and_preserve_configs(pilot, tmp_path, monkeypatch):
    calls = []

    def sbatch(command, **kwargs):
        assert command[0:2] == ["sbatch", "--parsable"]
        assert kwargs == {"check": True, "capture_output": True, "text": True}
        calls.append(command)
        runner = shlex.split(command[-1])
        config = Path(runner[runner.index("--config") + 1])
        dataset = runner[runner.index("--dataset") + 1]
        assert yaml.safe_load(config.read_text())["dataset"] == dataset
        assert Path(runner[runner.index("--output") + 1]).name == "00_cpdino_baseline"
        return SimpleNamespace(stdout=f"{100 + len(calls)};cluster\n")

    monkeypatch.setattr(phenotyping.subprocess, "run", sbatch)
    result = CliRunner().invoke(
        phenotyping.main,
        [
            "--config",
            str(pilot),
            "--submit",
            "--experiment",
            "plate_a",
            "--experiment",
            "plate_b",
            "--experiment",
            "plate_a",
            "--config-dir",
            str(tmp_path / "jobs"),
            "--datasets-root",
            str(tmp_path / "outputs"),
            "--workspace-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 2
    assert "plate_a: submitted job 101" in result.output
    assert "plate_b: submitted job 102" in result.output
    assert len(list((tmp_path / "jobs").glob("batch-*/*.yml"))) == 2


def test_submission_failure_exposes_successful_jobs_and_stops(pilot, tmp_path, monkeypatch):
    calls = []

    def sbatch(command, **kwargs):
        calls.append(command)
        if len(calls) == 2:
            raise subprocess.CalledProcessError(1, command, stderr="partition unavailable")
        return SimpleNamespace(stdout="12345\n")

    monkeypatch.setattr(phenotyping.subprocess, "run", sbatch)
    result = CliRunner().invoke(
        phenotyping.main,
        [
            "--config",
            str(pilot),
            "--submit",
            "--experiment",
            "plate_a",
            "--experiment",
            "plate_b",
            "--config-dir",
            str(tmp_path / "jobs"),
            "--datasets-root",
            str(tmp_path / "outputs"),
        ],
    )
    assert result.exit_code != 0
    assert "plate_a: submitted job 12345" in result.output
    assert "partition unavailable" in result.output
    assert len(calls) == 2


def test_duplicate_submission_plan_rejected(pilot, tmp_path, forbid_launch):
    run = phenotyping.plan_phenotyping_runs(pilot)[0]
    with pytest.raises(ValueError, match="unique config/dataset"):
        phenotyping.submit_phenotyping_runs([run, run], config_dir=tmp_path / "jobs", workspace=tmp_path)
    assert not (tmp_path / "jobs").exists()


def test_matrix_stage_is_opt_in_and_deduplicates_model_rows(pilot, tmp_path, forbid_launch):
    rows = [{"phenotyping_config": str(pilot), "datasets_root": str(tmp_path / "outputs")}] * 4
    assert phenotyping.plan_matrix_phenotyping(rows, stages=("predict", "eval")) == []
    runs = phenotyping.plan_matrix_phenotyping(rows, stages=("phenotype", "predict"))
    assert [run.dataset for run in runs] == ["plate_a", "plate_b"]
    assert not (tmp_path / "outputs").exists()
    assert phenotyping.plan_matrix_phenotyping([{"phenotyping_config": "/missing/file.yml"}], stages=("predict",)) == []
    with pytest.raises(ValueError, match="requires defaults.phenotyping_config"):
        phenotyping.plan_matrix_phenotyping([{}], stages=("phenotype",))


def test_conflicting_configs_for_same_dataset_rejected(pilot, tmp_path, forbid_launch):
    other = write_yaml(tmp_path / "other.yml", yaml.safe_load(pilot.read_text()))
    rows = [{"phenotyping_config": str(path)} for path in (pilot, other)]
    with pytest.raises(ValueError, match="Conflicting phenotyping configurations"):
        phenotyping.plan_matrix_phenotyping(rows, stages=("phenotype",))


def test_numbered_attempts_are_direct_atomic_and_append_only(pilot, tmp_path, monkeypatch):
    base = tmp_path / "outputs" / "plate_a" / "2-phenotyping" / "segmentations_membrane"
    old = base / "00_cpdino_baseline"
    old.mkdir(parents=True)
    sentinel = old / "completed.json"
    sentinel.write_text("immutable")
    (base / "04_other_attempt").mkdir()
    config = yaml.safe_load(pilot.read_text())
    config["datasets"] = {"plate_a": config["datasets"]["plate_a"]}
    config["datasets"]["plate_a"]["output"] = str(old)
    write_yaml(pilot, config)
    runs = phenotyping.plan_phenotyping_runs(pilot)
    assert runs[0].output_base == base
    assert runs[0].output == base / "05_cpdino_baseline"
    assert not runs[0].output.exists()
    # Simulate a competing submitter claiming the preview before submission.
    runs[0].output.mkdir()
    monkeypatch.setattr(phenotyping.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="111\n"))
    jobs = phenotyping.submit_phenotyping_runs(runs, config_dir=tmp_path / "jobs", workspace=tmp_path)
    assert jobs == {runs[0].key: "111"}
    (materialized,) = (tmp_path / "jobs").glob("batch-*/*.yml")
    actual = yaml.safe_load(materialized.read_text())
    assert Path(actual["output"]) == base / "06_cpdino_baseline"
    assert Path(actual["output"]).is_dir()
    assert sentinel.read_text() == "immutable"
    # Concurrent reservations must each obtain a fresh direct child.
    with ThreadPoolExecutor(max_workers=4) as pool:
        attempts = list(pool.map(phenotyping._allocate_attempt, [base] * 8))
    assert len(set(attempts)) == 8
    assert all(attempt.parent == base for attempt in attempts)


def test_slurm_gpu_defaults_threads_and_matrix_interpreter(pilot, tmp_path):
    interpreter = tmp_path / ".venv-phenotyping" / "bin" / "python"
    models = [{"phenotyping_config": str(pilot), "phenotyping_python": str(interpreter)}]
    run = phenotyping.plan_matrix_phenotyping(models, stages=("phenotype",))[0]
    command = phenotyping.build_sbatch_cmd(run, tmp_path / "job.yml", workspace=tmp_path)
    assert "--partition=gpu" in command
    assert "--gpus=1" in command
    assert "--mem=64G" in command
    assert shlex.split(command[-1])[:4] == ["env", "OPENBLAS_NUM_THREADS=4", "OMP_NUM_THREADS=4", str(interpreter)]


def test_core_config_roundtrip_and_collection_exclusions(pilot, tmp_path):
    from dynaclr.phenotyping.config import load_config

    config = yaml.safe_load(pilot.read_text())
    entry = {key: str(tmp_path / value) for key, value in config["datasets"]["plate_a"].items()}
    core = write_yaml(
        tmp_path / "single.yml",
        {
            "dataset": "plate_a",
            **entry,
            "channels": CHANNELS,
            "exclude_fovs": ["A/1/0"],
        },
    )
    collection = write_yaml(
        tmp_path / "collection.yml",
        {
            "name": "collection",
            "experiments": [
                {
                    "name": "plate_a",
                    "data_path": entry["focus_source"],
                    "tracks_path": entry["tracks"],
                    "exclude_fovs": ["A/1/0", "B/1/0"],
                },
                {
                    "name": "plate_b",
                    "data_path": "unused",
                    "tracks_path": "unused",
                },
            ],
        },
    )
    (run,) = phenotyping.plan_phenotyping_runs(core, collection=collection)
    materialized = write_yaml(tmp_path / "job.yml", run.config)
    parsed = load_config(materialized, dataset="plate_a", output=run.output)
    assert parsed.exclude_fovs == ("A/1/0", "B/1/0")
    assert parsed.fluorescence[0].marker == "SEC61B"


def test_missing_fluorescence_and_mixed_plate_fail(pilot, tmp_path):
    config = yaml.safe_load(pilot.read_text())
    config.pop("channels")
    write_yaml(pilot, config)
    with pytest.raises(ValueError, match="channels.fluorescence is required"):
        phenotyping.plan_phenotyping_runs(pilot)
    config["channels"] = CHANNELS
    config["datasets"] = {"plate_a": config["datasets"]["plate_a"]}
    write_yaml(pilot, config)
    collection = write_yaml(
        tmp_path / "collection.yml",
        {
            "name": "mixed",
            "experiments": [
                {
                    "name": "plate_a",
                    "data_path": "unused",
                    "tracks_path": "unused",
                    "channels": [
                        {"name": "GFP", "marker": "SEC61B", "wells": ["A/1"]},
                        {"name": "GFP", "marker": "TOMM20", "wells": ["B/1"]},
                    ],
                }
            ],
        },
    )
    with pytest.raises(ValueError, match="missing per-well marker TOMM20"):
        phenotyping.plan_phenotyping_runs(pilot, collection=collection)
    config["channels"] = {
        **CHANNELS,
        "fluorescence": [
            {"name": "GFP", "marker": "SEC61B", "role": "organelle"},
            {"name": "GFP", "marker": "TOMM20", "role": "organelle"},
        ],
    }
    write_yaml(pilot, config)
    (run,) = phenotyping.plan_phenotyping_runs(pilot, collection=collection)
    assert [channel["wells"] for channel in run.config["channels"]["fluorescence"]] == [["A/1"], ["B/1"]]
    from dynaclr.phenotyping.config import load_config

    parsed = load_config(write_yaml(tmp_path / "mixed-core.yml", run.config))
    assert len(parsed.fluorescence) == 2


def write_volume(store, depth=86, virtual_channels=True):
    store.mkdir(parents=True, exist_ok=True)
    (store / ".zattrs").write_text(json.dumps({"plate": {"wells": [{"path": "A/1"}]}}))
    well = store / "A/1"
    position = well / "0"
    array = position / "0"
    array.mkdir(parents=True)
    (well / ".zattrs").write_text(json.dumps({"well": {"images": [{"path": "0"}]}}))
    labels = ["membrane_prediction", "nuclei_prediction"] if virtual_channels else ["GFP"]
    (position / ".zattrs").write_text(
        json.dumps(
            {
                "omero": {"channels": [{"label": label} for label in labels]},
                "multiscales": [{"axes": [{"name": axis} for axis in "tczyx"], "datasets": [{"path": "0"}]}],
            }
        )
    )
    (array / ".zarray").write_text(json.dumps({"shape": [2, len(labels), depth, 32, 32]}))


def test_source_resolution_ignores_newer_mip_and_rejects_ambiguity(tmp_path):
    stage = tmp_path / "plate_a" / "4-assemble"
    exact = stage / "plate_a.zarr"
    mip = stage / "plate_a_mip.zarr"
    write_volume(exact)
    write_volume(mip, depth=1)
    os.utime(exact / ".zattrs", (1, 1))
    os.utime(mip / ".zattrs", (2, 2))
    assert phenotyping._resolve_source_assembly("plate_a", tmp_path, CHANNELS) == exact
    # Without the exact name, volume metadata is required and uniqueness enforced.
    renamed = stage / "plate_a_1.zarr"
    exact.rename(renamed)
    assert phenotyping._resolve_source_assembly("plate_a", tmp_path, CHANNELS) == renamed
    write_volume(stage / "raw-only.zarr", virtual_channels=False)
    assert phenotyping._resolve_source_assembly("plate_a", tmp_path, CHANNELS) == renamed
    write_volume(stage / "plate_a_2.zarr")
    with pytest.raises(ValueError, match="ambiguous source assemblies"):
        phenotyping._resolve_source_assembly("plate_a", tmp_path, CHANNELS)
