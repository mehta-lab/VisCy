"""Tests for airtable_utils.prepare — CSV-sidecar (csv_dir) wiring."""

from __future__ import annotations

from pathlib import Path

from airtable_utils.prepare import (
    PrepareConfig,
    PreprocessParams,
    QCParams,
    SlurmStageConfig,
    check_preprocessed,
    discover_zarr_stores,
    generate_preprocess_slurm,
    generate_qc_config,
    generate_qc_slurm,
)
from viscy_data.meta_csv import metadata_to_rows, write_meta_rows_csv


def test_prepare_config_default_csv_dir_is_none():
    """csv_dir defaults to None, so existing configs without it are unaffected."""
    cfg = PrepareConfig()
    assert cfg.csv_dir is None


def test_prepare_config_accepts_csv_dir(tmp_path):
    """csv_dir can be set to a path."""
    cfg = PrepareConfig(csv_dir=tmp_path)
    assert cfg.csv_dir == tmp_path


def test_prepare_config_default_jobs_dir_is_none():
    """jobs_dir defaults to None; batch-preprocess falls back to csv_dir."""
    cfg = PrepareConfig()
    assert cfg.jobs_dir is None


def test_check_preprocessed_csv_dir_present(tmp_path):
    """check_preprocessed reads the CSV sidecar when csv_dir is given."""
    store_path = "/data/exp.zarr"
    rows = metadata_to_rows(
        {"fov_statistics": {"mean": 1.0, "std": 0.1, "median": 1.0, "iqr": 0.2, "max": 5.0, "min": 0.0}},
        store_path=store_path,
        position_path="A/1/0",
        channel_name="Phase3D",
        field_name="normalization",
    )
    write_meta_rows_csv(tmp_path, store_path, rows)

    assert check_preprocessed(Path(store_path), csv_dir=tmp_path) is True


def test_check_preprocessed_csv_dir_absent(tmp_path):
    """check_preprocessed returns False (not an error) when no sidecar exists yet."""
    assert check_preprocessed(Path("/data/never_run.zarr"), csv_dir=tmp_path) is False


def test_check_preprocessed_csv_dir_ignores_other_field(tmp_path):
    """A sidecar with only focus_slice rows (no normalization) is not 'preprocessed'."""
    store_path = "/data/exp.zarr"
    rows = metadata_to_rows(
        {"fov_statistics": {"z_focus_mean": 4.0, "z_focus_std": 0.5}},
        store_path=store_path,
        position_path="A/1/0",
        channel_name="Phase3D",
        field_name="focus_slice",
    )
    write_meta_rows_csv(tmp_path, store_path, rows)

    assert check_preprocessed(Path(store_path), csv_dir=tmp_path) is False


def test_generate_qc_config_includes_csv_dir(tmp_path):
    """csv_dir is written into the qc_config.yml dict when given."""
    cfg = generate_qc_config(Path("/data/exp.zarr"), QCParams(), csv_dir=tmp_path)
    assert cfg["csv_dir"] == str(tmp_path)


def test_generate_qc_config_omits_csv_dir_by_default():
    """csv_dir is None in the dict when not given, matching QCConfig's default."""
    cfg = generate_qc_config(Path("/data/exp.zarr"), QCParams())
    assert cfg["csv_dir"] is None


def test_generate_preprocess_slurm_includes_csv_dir_flag(tmp_path):
    """--csv_dir is appended to the viscy preprocess invocation when csv_dir is set."""
    script = generate_preprocess_slurm(
        dataset_name="exp",
        vast_output_dir=tmp_path,
        vast_zarr_path=tmp_path / "exp.zarr",
        workspace_dir=Path("/workspace"),
        preprocess_params=PreprocessParams(),
        slurm_cfg=SlurmStageConfig(partition="cpu"),
        csv_dir=tmp_path / "csvs",
    )
    assert f'--csv_dir "{tmp_path / "csvs"}"' in script
    assert "viscy preprocess" in script


def test_generate_preprocess_slurm_omits_csv_dir_flag_by_default(tmp_path):
    """No --csv_dir flag when csv_dir is not given (existing behavior unchanged)."""
    script = generate_preprocess_slurm(
        dataset_name="exp",
        vast_output_dir=tmp_path,
        vast_zarr_path=tmp_path / "exp.zarr",
        workspace_dir=Path("/workspace"),
        preprocess_params=PreprocessParams(),
        slurm_cfg=SlurmStageConfig(partition="cpu"),
    )
    assert "--csv_dir" not in script


def test_discover_zarr_stores_finds_dataset_zarrs_and_skips_tracking(tmp_path):
    """discover_zarr_stores finds {name}/{name}.zarr but skips sibling tracking.zarr."""
    for name in ("dsB", "dsA"):
        (tmp_path / name / f"{name}.zarr").mkdir(parents=True)
        (tmp_path / name / "tracking.zarr").mkdir(parents=True)
    (tmp_path / "not_a_dataset").mkdir()
    (tmp_path / "not_a_dataset" / "notes.txt").write_text("")

    stores = discover_zarr_stores(tmp_path)

    assert stores == [tmp_path / "dsA" / "dsA.zarr", tmp_path / "dsB" / "dsB.zarr"]


def test_discover_zarr_stores_empty_root(tmp_path):
    """No dataset subdirectories -> empty list, not an error."""
    assert discover_zarr_stores(tmp_path) == []


def test_slurm_header_includes_qos_when_set(tmp_path):
    """--qos is appended to the SBATCH header when the SlurmStageConfig sets it."""
    script = generate_qc_slurm(
        dataset_name="exp",
        vast_output_dir=tmp_path,
        qc_config_path=tmp_path / "qc_config.yml",
        workspace_dir=Path("/workspace"),
        slurm_cfg=SlurmStageConfig(partition="gpu", qos="mid"),
    )
    assert "#SBATCH --qos=mid" in script


def test_slurm_header_omits_qos_by_default(tmp_path):
    """No --qos line when qos is not set (existing Bruno configs unaffected)."""
    script = generate_qc_slurm(
        dataset_name="exp",
        vast_output_dir=tmp_path,
        qc_config_path=tmp_path / "qc_config.yml",
        workspace_dir=Path("/workspace"),
        slurm_cfg=SlurmStageConfig(partition="gpu"),
    )
    assert "--qos" not in script
