# %%
"""Batch DynaCLR prediction config & SLURM script generator.

Generates prediction YAML configs and SLURM submission scripts for
multiple datasets, channels, and checkpoints. Automatically resolves
z_range from focus_slice metadata (computing it on the fly if missing)
and detects source channel names from the zarr.

Usage: run cells interactively or execute as a script.
"""

import subprocess
from pathlib import Path

from iohub import open_ome_zarr
from utils import (
    FOCUS_PARAMS,
    MODEL_2D_BAG_TIMEAWARE,  # noqa: F401
    MODEL_3D_BAG_TIMEAWARE,
    build_registry,
    extract_epoch,
    find_phenotyping_predictions_dir,
    generate_slurm_script,
    generate_yaml,
    get_z_range,
    print_registry_summary,
    resolve_channel_name,
    resolve_dataset_paths,
)

# %%
# ===========================================================================
# USER CONFIGURATION
# ===========================================================================

BASE_DIR = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics")

# Choose model template
MODEL = MODEL_3D_BAG_TIMEAWARE
# MODEL = MODEL_2D_BAG_TIMEAWARE

VERSION = "v1"

CHANNELS = ["phase", "organelle", "sensor"]

CHECKPOINTS = [
    "/hpc/projects/organelle_phenotyping/models/bag_of_channels/"
    "h2b_caax_tomm_sec61_g3bp1_sensor_phase/tb_logs/"
    "dynaclr3d_bag_channels_v1/version_2/checkpoints/"
    "epoch=40-step=44746.ckpt",
]

# Datasets to process. Set to [] to auto-discover from annotations_only.
DATASETS = [
    "2025_01_24_A549_G3BP1_DENV",
    "2024_11_07_A549_SEC61_DENV",
    "2025_01_28_A549_G3BP1_ZIKV_DENV",
    "2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV",
]

# Per-dataset channel keyword overrides.
# E.g., {"2025_04_10_...": {"organelle": "Cy5"}}
CHANNEL_OVERRIDES: dict[str, dict[str, str]] = {}

# Annotations directory (used for auto-discovery when DATASETS is empty).
ANNOTATIONS_DIR = Path("/hpc/projects/organelle_phenotyping/datasets/annotations")

# Set to True for a dry run (preview only, no files written).
DRY_RUN = False

# Set to True to overwrite existing config files. False to skip them.
OVERWRITE_FILES = True

# Set to True to submit all generated predict_all.sh scripts via sbatch.
SUBMIT_JOBS = True

# %%
# ===========================================================================
# Discovery & validation
# ===========================================================================

# Auto-discover datasets from annotations when DATASETS is empty
if not DATASETS:
    registry, skipped, annotations_only, predictions_only = build_registry(
        BASE_DIR, ANNOTATIONS_DIR, MODEL["name"], VERSION
    )
    print_registry_summary(registry, skipped, annotations_only, predictions_only)
    DATASETS = annotations_only
    print(f"\nAuto-discovered {len(DATASETS)} datasets missing predictions.\n")

print("## Batch Prediction Config Generator\n")
print(f"- **Model**: `{MODEL['name']}`")
print(f"- **Version**: `{VERSION}`")
print(f"- **Channels**: {CHANNELS}")
print(f"- **Checkpoints**: {len(CHECKPOINTS)}")
print(f"- **Datasets**: {len(DATASETS)}")
print(f"- **Dry run**: {DRY_RUN}\n")

validated: list[dict] = []
errors: list[dict] = []

for ds in DATASETS:
    try:
        paths = resolve_dataset_paths(ds, BASE_DIR, MODEL)
        print(f"Resolving {ds}...")

        # Read channel names from data zarr
        plate = open_ome_zarr(str(paths["data_path"]), mode="r")
        zarr_channels = list(plate.channel_names)
        plate.close()

        # Resolve channel names
        ds_overrides = CHANNEL_OVERRIDES.get(ds)
        available = {}
        for ch_type in CHANNELS:
            ch_name = resolve_channel_name(zarr_channels, ch_type, ds_overrides)
            if ch_name:
                available[ch_type] = ch_name
            else:
                print(f"  WARNING: channel '{ch_type}' not found in {ds}")

        # Resolve z_range (may compute focus on the fly)
        phase_ch = available.get("phase")
        z_range = get_z_range(paths["data_path"], MODEL, FOCUS_PARAMS, phase_channel=phase_ch)
        print(f"  z_range: {z_range}")

        validated.append(
            {
                "dataset": ds,
                "paths": paths,
                "z_range": z_range,
                "channels": available,
            }
        )

    except Exception as e:
        errors.append({"dataset": ds, "error": str(e)})
        print(f"  ERROR: {e}")

# %%
# ===========================================================================
# Summary before generation
# ===========================================================================

print("\n### Validated Datasets\n")
print("| Dataset | z_range | Channels | data_path |")
print("|---------|---------|----------|-----------|")
for v in validated:
    ch_str = ", ".join(sorted(v["channels"].keys()))
    print(f"| {v['dataset']} | {v['z_range']} | {ch_str} | `{v['paths']['data_path'].name}` |")

if errors:
    print("\n### Errors\n")
    print("| Dataset | Error |")
    print("|---------|-------|")
    for e in errors:
        print(f"| {e['dataset']} | {e['error']} |")

print(
    f"\n**Will generate**: {len(validated)} datasets "
    f"x {len(CHECKPOINTS)} checkpoints "
    f"= {len(validated) * len(CHECKPOINTS)} config sets"
)

# %%
# ===========================================================================
# Generate configs and scripts
# ===========================================================================

generated: list[dict] = []

for entry in validated:
    ds = entry["dataset"]
    paths = entry["paths"]
    z_range = entry["z_range"]
    channels = entry["channels"]

    output_dir = find_phenotyping_predictions_dir(BASE_DIR / ds, MODEL["name"], VERSION)

    for ckpt in CHECKPOINTS:
        epoch = extract_epoch(ckpt)
        suffix = ""
        files_written = []

        for ch_type, ch_name in channels.items():
            yml_content = generate_yaml(
                ds,
                paths["data_path"],
                paths["tracks_path"],
                MODEL,
                ch_type,
                ch_name,
                z_range,
                ckpt,
                output_dir,
                VERSION,
            )
            sh_content = generate_slurm_script(ch_type, output_dir, suffix=suffix)

            yml_path = output_dir / f"predict_{ch_type}{suffix}.yml"
            sh_path = output_dir / f"predict_{ch_type}{suffix}.sh"

            if not OVERWRITE_FILES and yml_path.exists():
                print(f"  Skipping {yml_path.name} (exists)")
                continue

            if not DRY_RUN:
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "slurm_out").mkdir(exist_ok=True)
                yml_path.write_text(yml_content)
                sh_path.write_text(sh_content)
                sh_path.chmod(0o755)

            files_written.append(
                {
                    "channel": ch_type,
                    "yml": yml_path,
                    "sh": sh_path,
                    "yml_content": yml_content,
                    "sh_content": sh_content,
                }
            )

        # predict_all.sh
        if files_written:
            run_all_lines = ["#!/bin/bash", ""]
            for f in files_written:
                run_all_lines.append(f"sbatch {f['sh']}")
            run_all_content = "\n".join(run_all_lines) + "\n"

            run_all_name = f"predict_all{suffix}.sh"
            run_all_path = output_dir / run_all_name
            if not DRY_RUN:
                run_all_path.write_text(run_all_content)
                run_all_path.chmod(0o755)

            generated.append(
                {
                    "dataset": ds,
                    "checkpoint": ckpt,
                    "epoch": epoch,
                    "output_dir": output_dir,
                    "files": files_written,
                }
            )

# %%
# ===========================================================================
# Generation summary
# ===========================================================================

action = "Generated" if not DRY_RUN else "Would generate (DRY RUN)"
print(f"\n## {action}\n")
print("| Dataset | Epoch | Channels | Output Dir |")
print("|---------|-------|----------|------------|")
for g in generated:
    ch_str = ", ".join(f["channel"] for f in g["files"])
    print(f"| {g['dataset']} | {g['epoch']} | {ch_str} | `{g['output_dir']}` |")

print("\n### Files\n")
for g in generated:
    print(f"**{g['dataset']}** (epoch {g['epoch']}):")
    for f in g["files"]:
        print(f"  - `{f['yml']}`")
        print(f"  - `{f['sh']}`")
    print(f"  - `{g['output_dir'] / 'predict_all.sh'}`")

if DRY_RUN and generated:
    print("\n### Preview (first config)\n")
    print("```yaml")
    print(generated[0]["files"][0]["yml_content"])
    print("```")
    print("\nSet `DRY_RUN = False` to write files.")

# %%
# ===========================================================================
# Submit SLURM jobs
# ===========================================================================

if SUBMIT_JOBS and not DRY_RUN and generated:
    print("\n## Submitting SLURM jobs\n")
    print("| Dataset | Script | Job ID |")
    print("|---------|--------|--------|")
    for g in generated:
        predict_all = g["output_dir"] / "predict_all.sh"
        if not predict_all.exists():
            print(f"| {g['dataset']} | `{predict_all}` | MISSING |")
            continue
        result = subprocess.run(
            ["bash", str(predict_all)],
            capture_output=True,
            text=True,
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            print(f"| {g['dataset']} | `{predict_all.name}` | ERROR: {result.stderr.strip()} |")
        else:
            for line in output.splitlines():
                print(f"| {g['dataset']} | `{predict_all.name}` | {line} |")
elif SUBMIT_JOBS and DRY_RUN:
    print("\n**SUBMIT_JOBS is True but DRY_RUN is also True -- skipping submission.**")

# %%
