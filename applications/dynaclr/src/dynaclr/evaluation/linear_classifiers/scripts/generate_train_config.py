# %%
"""Generate linear classifier training YAML configs.

For each valid (task, channel) combination, generates a config file
that pairs embedding zarr files with annotation CSVs across all
datasets that have both.
"""

from pathlib import Path

import yaml

from dynaclr.evaluation.linear_classifiers.src.utils import (
    CHANNELS,
    TASKS,
    build_registry,
    print_registry_summary,
)

# %%
# --- Configuration ---
embeddings_dir = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics")
annotations_dir = Path("/hpc/projects/organelle_phenotyping/datasets/annotations")
model = "DynaCLR-2D-Bag*Channels-timeaware"
version = "v3"
output_dir = Path("/hpc/projects/organelle_phenotyping/models/linear_classifiers/configs")

# %%
# --- Discover datasets ---
registry, skipped, annotations_only, predictions_only = build_registry(embeddings_dir, annotations_dir, model, version)
print_registry_summary(registry, skipped, annotations_only, predictions_only)

# %%
# --- Generate configs for each task x channel ---
embedding_model_name = model.replace("*", "")
embedding_model_version = version
generated: list[dict] = []

for task in TASKS:
    for channel in CHANNELS:
        datasets_for_combo = []
        for entry in registry:
            if task in entry["available_tasks"] and channel in entry["channel_zarrs"]:
                datasets_for_combo.append(
                    {
                        "embeddings": str(entry["channel_zarrs"][channel]),
                        "annotations": str(entry["annotations_csv"]),
                    }
                )

        if not datasets_for_combo:
            continue

        config = {
            "task": task,
            "input_channel": channel,
            "embedding_model_name": embedding_model_name,
            "embedding_model_version": embedding_model_version,
            "train_datasets": datasets_for_combo,
            "use_scaling": True,
            "use_pca": False,
            "n_pca_components": None,
            "max_iter": 1000,
            "class_weight": "balanced",
            "solver": "liblinear",
            "split_train_data": 0.8,
            "random_seed": 42,
            "wandb_entity": None,
            "wandb_tags": [],
        }

        filename = f"{task}_{channel}.yaml"
        generated.append(
            {
                "task": task,
                "channel": channel,
                "n_datasets": len(datasets_for_combo),
                "filename": filename,
                "config": config,
            }
        )

# %%
# --- Print generation summary ---
print(f"\n## Generated Configs ({len(generated)} total)\n")
print("| Task | Channel | Datasets | File |")
print("|------|---------|----------|------|")
for entry in generated:
    print(f"| {entry['task']} | {entry['channel']} | {entry['n_datasets']} | `{entry['filename']}` |")

# %%
# --- Write YAML configs ---
output_dir.mkdir(parents=True, exist_ok=True)
for entry in generated:
    out_path = output_dir / entry["filename"]
    with open(out_path, "w") as f:
        yaml.dump(entry["config"], f, default_flow_style=False, sort_keys=False)
    print(f"Wrote {out_path}")

# %%
