# %%
"""Generate prediction .sh/.yml scripts for datasets missing embeddings.

Uses an existing dataset's prediction configs as a template, swaps in the
target dataset name, and enforces a single checkpoint across all datasets.
"""

import re
from glob import glob
from pathlib import Path

from natsort import natsorted

from dynaclr.evaluation.linear_classifiers.src.utils import (
    CHANNELS,
    build_registry,
    print_registry_summary,
)

# %%
# --- Configuration ---
embeddings_dir = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics")
annotations_dir = Path("/hpc/projects/organelle_phenotyping/datasets/annotations")
model = "DynaCLR-2D-Bag*Channels-timeaware"
version = "v3"
ckpt_path = (
    "/hpc/projects/organelle_phenotyping/models/"
    "SEC61_TOMM20_G3BP1_Sensor/time_interval/dynaclr_gfp_rfp_Ph/"
    "organelle_sensor_phase_maxproj_ver3_150epochs/saved_checkpoints/"
    "epoch=104-step=53760.ckpt"
)

# %%
# --- Discover datasets and gaps ---
registry, skipped, annotations_only, predictions_only = build_registry(embeddings_dir, annotations_dir, model, version)
print_registry_summary(registry, skipped, annotations_only, predictions_only)

# %%
# --- Pick reference dataset ---
if not registry:
    raise RuntimeError("No reference dataset found with both predictions and annotations.")

reference_dataset = registry[0]["dataset"]
reference_pred_dir = registry[0]["predictions_dir"]
reference_model_dir = reference_pred_dir.parent.name

print("\n## Prediction Script Generation\n")
print(f"- Reference dataset: `{reference_dataset}`")
print(f"- Reference dir: `{reference_pred_dir}`")
print(f"- Checkpoint: `{ckpt_path}`\n")

# %%
# --- Generate scripts for each dataset missing predictions ---
prediction_scripts_generated: list[dict] = []
generation_skipped: list[dict] = []

for target_dataset in annotations_only:
    target_base = embeddings_dir / target_dataset
    if not target_base.is_dir():
        generation_skipped.append({"dataset": target_dataset, "reason": "No directory in embeddings_dir"})
        continue

    phenotyping_matches = natsorted(glob(str(target_base / "*phenotyping*")))
    if not phenotyping_matches:
        generation_skipped.append({"dataset": target_dataset, "reason": "No *phenotyping* directory"})
        continue
    phenotyping_dir = Path(phenotyping_matches[0])

    # Find existing predictions parent or default to "predictions"
    pred_parent_matches = natsorted(glob(str(phenotyping_dir / "*prediction*")))
    pred_parent = Path(pred_parent_matches[0]) if pred_parent_matches else phenotyping_dir / "predictions"
    target_pred_dir = pred_parent / reference_model_dir / version

    # Verify data_path and tracks_path exist
    data_path_matches = natsorted(glob(str(phenotyping_dir / "train-test" / f"{target_dataset}*.zarr")))
    tracks_path_matches = natsorted(
        glob(str(target_base / "1-preprocess" / "label-free" / "3-track" / f"{target_dataset}*cropped.zarr"))
    )

    if not data_path_matches:
        generation_skipped.append({"dataset": target_dataset, "reason": "No train-test zarr found"})
        continue
    if not tracks_path_matches:
        generation_skipped.append({"dataset": target_dataset, "reason": "No tracking zarr found"})
        continue

    generated_files = []
    for channel in CHANNELS:
        ref_yml = reference_pred_dir / f"predict_{channel}.yml"
        ref_sh = reference_pred_dir / f"predict_{channel}.sh"

        if not ref_yml.exists() or not ref_sh.exists():
            continue

        # Swap dataset name in all paths
        new_yml = ref_yml.read_text().replace(reference_dataset, target_dataset)
        new_sh = ref_sh.read_text().replace(reference_dataset, target_dataset)

        # Enforce the configured checkpoint
        new_yml = re.sub(r"(?m)^ckpt_path:.*$", f"ckpt_path: {ckpt_path}", new_yml)

        generated_files.append(
            {
                "channel": channel,
                "yml_path": target_pred_dir / f"predict_{channel}.yml",
                "yml_content": new_yml,
                "sh_path": target_pred_dir / f"predict_{channel}.sh",
                "sh_content": new_sh,
            }
        )

    if generated_files:
        prediction_scripts_generated.append(
            {
                "dataset": target_dataset,
                "pred_dir": target_pred_dir,
                "files": generated_files,
            }
        )

# %%
# --- Print summary ---
if prediction_scripts_generated:
    print("### Will Generate\n")
    print("| Dataset | Prediction Dir | Channels |")
    print("|---------|---------------|----------|")
    for entry in prediction_scripts_generated:
        channels_str = ", ".join(f["channel"] for f in entry["files"])
        print(f"| {entry['dataset']} | `{entry['pred_dir']}` | {channels_str} |")
else:
    print("No datasets need prediction scripts generated.")

if generation_skipped:
    print("\n### Cannot Generate\n")
    print("| Dataset | Reason |")
    print("|---------|--------|")
    for s in generation_skipped:
        print(f"| {s['dataset']} | {s['reason']} |")

# %%
# --- Write prediction scripts and run_all.sh ---
for entry in prediction_scripts_generated:
    pred_dir = entry["pred_dir"]
    pred_dir.mkdir(parents=True, exist_ok=True)
    (pred_dir / "slurm_out").mkdir(exist_ok=True)

    sh_names = []
    for f in entry["files"]:
        f["yml_path"].write_text(f["yml_content"])
        f["sh_path"].write_text(f["sh_content"])
        f["sh_path"].chmod(0o755)
        sh_names.append(f["sh_path"].name)

    # Generate run_all.sh
    run_all_path = pred_dir / "run_all.sh"
    run_all_lines = ["#!/bin/bash", ""]
    for sh_name in sh_names:
        run_all_lines.append(f"sbatch {sh_name}")
    run_all_content = "\n".join(run_all_lines) + "\n"
    run_all_path.write_text(run_all_content)
    run_all_path.chmod(0o755)

    print(f"Wrote {entry['dataset']} -> {pred_dir}")
    for sh_name in sh_names:
        print(f"  {sh_name}")
    print("  run_all.sh")
