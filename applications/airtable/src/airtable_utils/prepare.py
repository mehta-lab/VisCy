"""Config-driven dataset preparation: NFS -> VAST rechunked zarr v3."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from textwrap import dedent

import yaml
from iohub import open_ome_zarr
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic config models
# ---------------------------------------------------------------------------


class ConcatenateConfig(BaseModel):
    """Parameters for biahub concatenate."""

    channel_names: list[str] | None = None
    chunks_czyx: list[int] = [1, 16, 256, 256]
    shards_ratio: list[int] = [1, 1, 8, 8, 8]
    output_ome_zarr_version: str = "0.5"
    conda_env: str = "biahub"
    sbatch_overrides: dict[str, str] | None = None


class QCParams(BaseModel):
    """Focus-slice QC parameters."""

    channel_names: list[str] = ["Phase3D"]
    NA_det: float = 1.35
    lambda_ill: float = 0.450
    pixel_size: float = 0.1494
    midband_fractions: tuple[float, float] = (0.125, 0.25)
    device: str = "cuda"
    num_workers: int = 16


class PreprocessParams(BaseModel):
    """Normalization preprocessing parameters."""

    channel_names: int | list[str] = -1
    num_workers: int = 48
    block_size: int = 32


class SlurmStageConfig(BaseModel):
    """SLURM resource settings for one job stage."""

    partition: str
    cpus_per_task: int = 24
    mem_per_cpu: str = "4G"
    time: str = "06:00:00"
    gres: str | None = None
    constraint: str | None = None


class SlurmConfig(BaseModel):
    """SLURM settings for QC and preprocess stages (separate jobs).

    The concatenation stage is not a SLURM job — ``biahub concatenate``
    submits its own SLURM jobs internally via submitit.
    """

    qc: SlurmStageConfig = Field(
        default_factory=lambda: SlurmStageConfig(
            partition="gpu",
            gres="gpu:1",
            cpus_per_task=16,
            mem_per_cpu="4G",
            time="00:30:00",
        )
    )
    preprocess: SlurmStageConfig = Field(
        default_factory=lambda: SlurmStageConfig(
            partition="preempted",
            cpus_per_task=16,
            mem_per_cpu="4G",
            time="04:00:00",
        )
    )


class PrepareConfig(BaseModel):
    """Top-level prepare pipeline configuration."""

    nfs_root: Path = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics")
    vast_root: Path = Path("/hpc/projects/organelle_phenotyping/datasets")
    workspace_dir: Path = Path("/hpc/mydata/eduardo.hirata/repos/viscy")
    concatenate: ConcatenateConfig = Field(default_factory=ConcatenateConfig)
    qc: QCParams = Field(default_factory=QCParams)
    preprocess: PreprocessParams = Field(default_factory=PreprocessParams)
    slurm: SlurmConfig = Field(default_factory=SlurmConfig)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def resolve_nfs_paths(dataset_name: str, nfs_root: Path) -> dict[str, Path]:
    """Return NFS zarr and tracking paths for a dataset.

    Parameters
    ----------
    dataset_name : str
        Dataset identifier, e.g. ``"2025_01_22_A549_G3BP1_ZIKV_DENV"``.
    nfs_root : Path
        Root of organelle_dynamics on NFS.

    Returns
    -------
    dict[str, Path]
        Keys: ``zarr``, ``tracking``.

    Raises
    ------
    FileNotFoundError
        If the assembled zarr does not exist on NFS.
    """
    zarr_path = nfs_root / dataset_name / "2-assemble" / f"{dataset_name}.zarr"
    tracking_path = nfs_root / dataset_name / "1-preprocess" / "label-free" / "3-track" / f"{dataset_name}_cropped.zarr"
    if not zarr_path.exists():
        raise FileNotFoundError(f"NFS zarr not found: {zarr_path}")
    return {"zarr": zarr_path, "tracking": tracking_path}


def resolve_vast_paths(dataset_name: str, vast_root: Path) -> dict[str, Path]:
    """Return expected VAST output paths for a dataset.

    Parameters
    ----------
    dataset_name : str
        Dataset identifier.
    vast_root : Path
        Root of datasets directory on VAST.

    Returns
    -------
    dict[str, Path]
        Keys: ``output_dir``, ``zarr``, ``tracking``.
    """
    output_dir = vast_root / dataset_name
    return {
        "output_dir": output_dir,
        "zarr": output_dir / f"{dataset_name}.zarr",
        "tracking": output_dir / "tracking.zarr",
    }


# ---------------------------------------------------------------------------
# Zarr version validation
# ---------------------------------------------------------------------------


def check_zarr_version(zarr_path: Path) -> dict[str, int | str | None]:
    """Check zarr format version and OME-Zarr version of an existing store.

    Parameters
    ----------
    zarr_path : Path
        Path to the zarr store root.

    Returns
    -------
    dict[str, int | str | None]
        Keys: ``zarr_format`` (2, 3, or None), ``ome_version`` (e.g. "0.5" or None).
    """
    result: dict[str, int | str | None] = {"zarr_format": None, "ome_version": None}

    zarr_json = zarr_path / "zarr.json"
    zgroup = zarr_path / ".zgroup"

    if zarr_json.exists():
        with open(zarr_json) as f:
            meta = json.load(f)
        result["zarr_format"] = meta.get("zarr_format", 3)
        ome = meta.get("attributes", {}).get("ome", {})
        result["ome_version"] = ome.get("version")
    elif zgroup.exists():
        with open(zgroup) as f:
            meta = json.load(f)
        result["zarr_format"] = meta.get("zarr_format", 2)
        zattrs = zarr_path / ".zattrs"
        if zattrs.exists():
            with open(zattrs) as f:
                attrs = json.load(f)
            result["ome_version"] = attrs.get("plate", {}).get("version")

    return result


def check_preprocessed(zarr_path: Path) -> bool:
    """Check if normalization metadata has been written to the zarr store.

    Parameters
    ----------
    zarr_path : Path
        Path to the zarr store root.

    Returns
    -------
    bool
        True if normalization stats are present.
    """
    zarr_json = zarr_path / "zarr.json"
    zattrs = zarr_path / ".zattrs"

    if zarr_json.exists():
        with open(zarr_json) as f:
            meta = json.load(f)
        return "normalization" in meta.get("attributes", {})
    elif zattrs.exists():
        with open(zattrs) as f:
            attrs = json.load(f)
        return "normalization" in attrs

    return False


# ---------------------------------------------------------------------------
# Discovery (reads NFS zarr via iohub)
# ---------------------------------------------------------------------------


def discover_wells(nfs_zarr_path: Path) -> list[str]:
    """Enumerate well paths from an NFS OME-Zarr plate.

    Returns well-level paths (e.g. ``"B/1"``) not full position paths.
    The ``crop_concat.yml`` format expects ``{zarr}/{well}/*`` globs
    so that biahub concatenate can discover positions within each well.

    Parameters
    ----------
    nfs_zarr_path : Path
        Path to the assembled zarr on NFS.

    Returns
    -------
    list[str]
        Sorted well paths like ``["A/1", "B/1", "C/2"]``.
    """
    wells: list[str] = []
    with open_ome_zarr(str(nfs_zarr_path), mode="r") as plate:
        for pos_path, _pos in plate.positions():
            # pos_path is like "A/1/000000" — extract well as "A/1"
            well = "/".join(pos_path.split("/")[:2])
            if well not in wells:
                wells.append(well)
    return sorted(wells)


def discover_channels(nfs_zarr_path: Path) -> list[str]:
    """Read channel names from an NFS OME-Zarr plate.

    Parameters
    ----------
    nfs_zarr_path : Path
        Path to the assembled zarr on NFS.

    Returns
    -------
    list[str]
        Channel names, e.g. ``["Phase3D", "raw GFP EX488 EM525-45", ...]``.
    """
    with open_ome_zarr(str(nfs_zarr_path), mode="r") as plate:
        return list(plate.channel_names)


RAW_CHANNEL_PREFIXES = ("Phase3D", "raw ")


def filter_raw_channels(channel_names: list[str]) -> list[str]:
    """Filter to only raw imaging channels (Phase3D and raw fluorescence).

    Excludes virtual stains (``nuclei_prediction``, ``membrane_prediction``),
    deconvolved channels (``GFP EX488 ...`` without ``raw`` prefix), and
    other derived channels (``BF``).

    Parameters
    ----------
    channel_names : list[str]
        All channel names from the zarr.

    Returns
    -------
    list[str]
        Only channels starting with ``"Phase3D"`` or ``"raw "``.
    """
    return [ch for ch in channel_names if ch.startswith(RAW_CHANNEL_PREFIXES)]


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------


def generate_crop_concat_config(
    nfs_zarr_path: Path,
    wells: list[str],
    channel_names: list[str],
    concat_cfg: ConcatenateConfig,
) -> dict:
    """Build a crop_concat.yml dict for biahub concatenate.

    Parameters
    ----------
    nfs_zarr_path : Path
        Path to the source zarr on NFS.
    wells : list[str]
        Well paths like ``["A/1", "B/2"]`` (row/col level).
        Each becomes ``"{zarr}/{well}/*"`` so biahub globs positions within.
    channel_names : list[str]
        Channel names (repeated once per well entry).
    concat_cfg : ConcatenateConfig
        Concatenation parameters.

    Returns
    -------
    dict
        Config dict ready to write as YAML.
    """
    concat_data_paths = [f"{nfs_zarr_path}/{well}/*" for well in wells]
    return {
        "concat_data_paths": concat_data_paths,
        "time_indices": "all",
        "channel_names": [channel_names] * len(wells),
        "X_slice": "all",
        "Y_slice": "all",
        "Z_slice": "all",
        "chunks_czyx": concat_cfg.chunks_czyx,
        "shards_ratio": concat_cfg.shards_ratio,
        "output_ome_zarr_version": concat_cfg.output_ome_zarr_version,
    }


def generate_qc_config(data_path: Path, qc_params: QCParams) -> dict:
    """Build a QC config dict compatible with ``qc run -c``.

    Parameters
    ----------
    data_path : Path
        Path to the VAST zarr (target of QC).
    qc_params : QCParams
        Focus-slice QC parameters.

    Returns
    -------
    dict
        Config dict ready to write as YAML.
    """
    return {
        "data_path": str(data_path),
        "num_workers": qc_params.num_workers,
        "focus_slice": {
            "channel_names": qc_params.channel_names,
            "NA_det": qc_params.NA_det,
            "lambda_ill": qc_params.lambda_ill,
            "pixel_size": qc_params.pixel_size,
            "midband_fractions": list(qc_params.midband_fractions),
            "device": qc_params.device,
        },
    }


def write_yaml(config: dict, output_path: Path) -> None:
    """Write a dict to a YAML file.

    Parameters
    ----------
    config : dict
        Config to serialize.
    output_path : Path
        Destination file path.
    """

    # Use a Dumper subclass that avoids YAML anchors/aliases for repeated
    # lists. Patching yaml.Dumper directly leaks into every other yaml.dump
    # in the same Python process.
    class _NoAliasDumper(yaml.Dumper):
        def ignore_aliases(self, data: object) -> bool:
            return True

    with open(output_path, "w") as f:
        yaml.dump(config, f, Dumper=_NoAliasDumper, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# SLURM script generation
# ---------------------------------------------------------------------------


def _slurm_header(job_name: str, output_dir: Path, cfg: SlurmStageConfig) -> str:
    """Build SBATCH header lines."""
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=1",
        f"#SBATCH --partition={cfg.partition}",
        f"#SBATCH --cpus-per-task={cfg.cpus_per_task}",
        f"#SBATCH --mem-per-cpu={cfg.mem_per_cpu}",
        f"#SBATCH --time={cfg.time}",
        f"#SBATCH --output={output_dir}/slurm_{job_name}_%j.out",
    ]
    if cfg.gres:
        lines.append(f"#SBATCH --gres={cfg.gres}")
    if cfg.constraint:
        lines.append(f'#SBATCH --constraint="{cfg.constraint}"')
    return "\n".join(lines)


def generate_sbatch_override_file(overrides: dict[str, str]) -> str:
    """Generate content for a biahub sbatch override file.

    Parameters
    ----------
    overrides : dict[str, str]
        SLURM directive keys and values, e.g.
        ``{"partition": "preempted", "mem-per-cpu": "16G"}``.

    Returns
    -------
    str
        File content with ``#SBATCH`` lines.
    """
    lines = ["#!/bin/bash"]
    for key, value in overrides.items():
        lines.append(f"#SBATCH --{key}={value}")
    return "\n".join(lines) + "\n"


def generate_concatenate_script(
    crop_concat_path: Path,
    vast_zarr_path: Path,
    nfs_tracking_path: Path,
    vast_tracking_path: Path,
    conda_env: str,
    sbatch_override_path: Path | None = None,
) -> str:
    """Generate a bash script for biahub concatenate + tracking copy.

    This is NOT a SLURM script. ``biahub concatenate`` submits its own
    SLURM jobs internally via submitit. The ``-m`` flag makes it block
    until those jobs complete. After concatenation, tracking is rsynced.

    Parameters
    ----------
    crop_concat_path : Path
        Path to the generated crop_concat.yml.
    vast_zarr_path : Path
        Target zarr output path.
    nfs_tracking_path : Path
        Source tracking zarr on NFS.
    vast_tracking_path : Path
        Target tracking zarr on VAST.
    conda_env : str
        Conda environment name for biahub.
    sbatch_override_path : Path or None
        Path to sbatch override file for biahub's internal SLURM jobs.

    Returns
    -------
    str
        Bash script content.
    """
    # Build the biahub command as a single line to avoid conda run
    # swallowing backslash continuations.
    cmd_parts = [
        f"conda run -n {conda_env} biahub concatenate",
        f'-c "{crop_concat_path}"',
        f'-o "{vast_zarr_path}"',
        "-m",
    ]
    if sbatch_override_path:
        cmd_parts.append(f'-sb "{sbatch_override_path}"')
    biahub_cmd = " ".join(cmd_parts)

    return dedent(f"""\
        #!/bin/bash
        set -euo pipefail

        echo "=== Step 1: biahub concatenate (submits SLURM jobs via submitit) ==="
        {biahub_cmd}
        echo "Concatenation complete."

        echo "=== Step 2: Copy tracking zarr ==="
        if [ -d "{nfs_tracking_path}" ]; then
            rsync -a --copy-links "{nfs_tracking_path}/" "{vast_tracking_path}/"
            echo "Tracking copy complete."
        else
            echo "WARNING: NFS tracking zarr not found at {nfs_tracking_path}, skipping."
        fi
    """)


def generate_qc_slurm(
    dataset_name: str,
    vast_output_dir: Path,
    qc_config_path: Path,
    workspace_dir: Path,
    slurm_cfg: SlurmStageConfig,
) -> str:
    """Generate SLURM script for focus-slice QC (needs GPU).

    Parameters
    ----------
    dataset_name : str
        Dataset identifier (used for job name).
    vast_output_dir : Path
        Output directory on VAST.
    qc_config_path : Path
        Path to the generated qc_config.yml.
    workspace_dir : Path
        Path to the viscy repo root.
    slurm_cfg : SlurmStageConfig
        SLURM resource parameters.

    Returns
    -------
    str
        Complete SLURM script content.
    """
    header = _slurm_header(f"qc_{dataset_name}", vast_output_dir, slurm_cfg)
    body = dedent(f"""\

        export PYTHONNOUSERSITE=1

        echo "=== QC: focus slice detection ==="
        uv run --project "{workspace_dir}" --package qc \
            qc run -c "{qc_config_path}"
        echo "QC complete."
    """)
    return header + "\n" + body


def generate_preprocess_slurm(
    dataset_name: str,
    vast_output_dir: Path,
    vast_zarr_path: Path,
    workspace_dir: Path,
    preprocess_params: PreprocessParams,
    slurm_cfg: SlurmStageConfig,
) -> str:
    """Generate SLURM script for normalization preprocessing (CPU only).

    Parameters
    ----------
    dataset_name : str
        Dataset identifier (used for job name).
    vast_output_dir : Path
        Output directory on VAST.
    vast_zarr_path : Path
        Path to the rechunked zarr on VAST.
    workspace_dir : Path
        Path to the viscy repo root.
    preprocess_params : PreprocessParams
        Normalization preprocessing parameters.
    slurm_cfg : SlurmStageConfig
        SLURM resource parameters.

    Returns
    -------
    str
        Complete SLURM script content.
    """
    header = _slurm_header(f"preprocess_{dataset_name}", vast_output_dir, slurm_cfg)

    ch_arg = preprocess_params.channel_names
    if isinstance(ch_arg, int):
        ch_flag = f"--channel_names={ch_arg}"
    else:
        ch_flag = " ".join(f"--channel_names={c}" for c in ch_arg)

    body = dedent(f"""\

        export PYTHONNOUSERSITE=1

        echo "=== Preprocess: normalization stats ==="
        echo "Data: {vast_zarr_path}"
        uv run --project "{workspace_dir}" --package dynaclr \
            viscy preprocess --data_path "{vast_zarr_path}" \
            {ch_flag} --num_workers {preprocess_params.num_workers} \
            --block_size {preprocess_params.block_size}
        echo "Preprocess complete."
    """)
    return header + "\n" + body


# ---------------------------------------------------------------------------
# Status check
# ---------------------------------------------------------------------------


def check_dataset_status(dataset_name: str, nfs_root: Path, vast_root: Path) -> dict[str, str]:
    """Check existence and version info for a dataset across NFS and VAST.

    Parameters
    ----------
    dataset_name : str
        Dataset identifier.
    nfs_root : Path
        NFS root directory.
    vast_root : Path
        VAST root directory.

    Returns
    -------
    dict[str, str]
        Status fields for the dataset.
    """
    nfs_zarr = nfs_root / dataset_name / "2-assemble" / f"{dataset_name}.zarr"
    vast = resolve_vast_paths(dataset_name, vast_root)

    nfs_exists = nfs_zarr.exists()
    vast_zarr_exists = vast["zarr"].exists()
    vast_tracking_exists = vast["tracking"].exists()

    zarr_fmt: str = "-"
    ome_ver: str = "-"
    preprocessed: str = "-"

    if vast_zarr_exists:
        ver = check_zarr_version(vast["zarr"])
        zarr_fmt = str(ver["zarr_format"]) if ver["zarr_format"] else "?"
        ome_ver = str(ver["ome_version"]) if ver["ome_version"] else "?"
        preprocessed = "yes" if check_preprocessed(vast["zarr"]) else "no"

    return {
        "dataset": dataset_name,
        "nfs": "yes" if nfs_exists else "no",
        "vast_zarr": "yes" if vast_zarr_exists else "no",
        "zarr_version": zarr_fmt,
        "ome_version": ome_ver,
        "tracking": "yes" if vast_tracking_exists else "no",
        "preprocessed": preprocessed,
    }


def format_status_table(rows: list[dict[str, str]]) -> str:
    """Format dataset status rows as a markdown table.

    Parameters
    ----------
    rows : list[dict[str, str]]
        Each dict from :func:`check_dataset_status`.

    Returns
    -------
    str
        Markdown table string.
    """
    headers = [
        "dataset",
        "nfs",
        "vast_zarr",
        "zarr_version",
        "ome_version",
        "tracking",
        "preprocessed",
    ]
    col_widths = {h: max(len(h), *(len(r[h]) for r in rows)) for h in headers}

    header_line = "| " + " | ".join(h.ljust(col_widths[h]) for h in headers) + " |"
    sep_line = "| " + " | ".join("-" * col_widths[h] for h in headers) + " |"
    data_lines = ["| " + " | ".join(r[h].ljust(col_widths[h]) for h in headers) + " |" for r in rows]
    return "\n".join([header_line, sep_line, *data_lines])
