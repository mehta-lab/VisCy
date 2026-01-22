"""Example usage of W&B Model Registry for curating and loading models.

This demonstrates Part 2 of the W&B integration:
- Registering trained models as artifacts
- Loading models from the registry
- Querying available models
- Full lineage: Collections → Training Run → Model Artifact
"""

# %%
# =============================================================================
# Part 1: After Training - Register a "Blessed" Model
# =============================================================================

from viscy.airtable.register_model import register_model

# After training completes, find the best checkpoint
checkpoint_path = "logs/wandb/run-20260107-154117/checkpoints/epoch=50-step=2550.ckpt"
wandb_run_id = "20260107-154117"  # Get from W&B UI or training output

# Register the model
artifact_url = register_model(
    checkpoint_path=checkpoint_path,
    model_name="contrastive-a549-sec61",
    model_type="contrastive",
    version="v1",
    aliases=["production", "best"],
    wandb_run_id=wandb_run_id,
    wandb_project="viscy-model-registry",
    shared_dir="/hpc/models/shared",
    description="Contrastive model trained on A549 SEC61 DENV wells B1-B2, val_loss=0.152",
    metadata={
        "val_loss": 0.152,
        "collection_name": "2024_11_07_A549_SEC61_DENV_wells_B1_B2",
        "collection_version": "0.0.1",
        "backbone": "convnext_tiny",
        "embedding_dim": 768,
    },
)

print("\n✓ Model registered successfully!")
print(f"  View in W&B: {artifact_url}")
print("  Checkpoint: /hpc/models/shared/contrastive/contrastive-a549-sec61-v1.ckpt")

# %%
# =============================================================================
# Part 2: Discovery - List Available Models
# =============================================================================

from viscy.airtable.register_model import list_registered_models

# List all registered models
all_models = list_registered_models(wandb_project="viscy-model-registry")

print("\n=== All Registered Models ===")
for model in all_models:
    print(f"{model['name']}:{model['version']}")
    print(f"  Type: {model['model_type']}")
    print(f"  Aliases: {model['aliases']}")
    print(f"  Description: {model['description']}")
    print(f"  Checkpoint: {model['checkpoint_path']}")
    print()

# %%
# Filter by model type
contrastive_models = list_registered_models(
    wandb_project="viscy-model-registry", model_type="contrastive"
)

print("\n=== Contrastive Models ===")
for model in contrastive_models:
    metadata = model["metadata"]
    print(f"{model['name']}:{model['version']}")
    print(f"  Val Loss: {metadata.get('val_loss', 'N/A')}")
    print(f"  Collections: {metadata.get('collection_name', 'N/A')}")
    print()

# %%
# Find production models
production_models = [m for m in all_models if "production" in m["aliases"]]

print("\n=== Production Models ===")
for model in production_models:
    print(f"- {model['name']} ({model['model_type']})")

# %%
# =============================================================================
# Part 3: Loading Models - Use Registered Models in Analysis/Inference
# =============================================================================

from viscy.airtable.register_model import load_model_from_registry
from viscy.representation.engine import ContrastiveModule

# Load production model by alias
model = load_model_from_registry(
    model_name="contrastive-a549-sec61",
    version="production",  # Can also use "v1", "latest", "best"
    wandb_project="viscy-model-registry",
    model_class=ContrastiveModule,
)

print("\n=== Model Loaded ===")
print(f"Model type: {type(model).__name__}")
print(f"Model in eval mode: {not model.training}")

# %%
# Use model for inference
import torch

# Create dummy input (replace with real data)
dummy_batch = torch.randn(2, 1, 5, 160, 160)  # [batch, channels, z, y, x]

with torch.no_grad():
    embeddings = model(dummy_batch)

print("\n✓ Inference successful!")
print(f"  Input shape: {dummy_batch.shape}")
print(f"  Embedding shape: {embeddings.shape}")

# %%
# =============================================================================
# Part 4: Full Lineage - From Collections to Model
# =============================================================================

import wandb

# Get model artifact
api = wandb.Api()
artifact = api.artifact("viscy-model-registry/contrastive-a549-sec61:production")

print("\n=== Model Lineage ===")
print(f"Model: {artifact.name}:{artifact.version}")
print(f"Description: {artifact.description}")
print()

# Get training run (lineage)
training_run_id = artifact.metadata.get("training_run_id")
if training_run_id:
    # NOTE: You may need to adjust project name
    try:
        training_run = api.run(f"eduardo-hirata/viscy-experiments/{training_run_id}")
        print(f"Training Run: {training_run.name}")
        print(f"  URL: {training_run.url}")
        print(f"  Metrics: val_loss={training_run.summary.get('loss/val', 'N/A')}")
        print()
    except Exception as e:
        print(f"Could not fetch training run: {e}")

# Get collection (from training run config or artifact metadata)
collection_name = artifact.metadata.get("collection_name")
collection_version = artifact.metadata.get("collection_version")

if collection_name and collection_version:
    print(f"Collections: {collection_name} v{collection_version}")

    # You can now fetch the full collection from Airtable
    from viscy.airtable.database import AirtableManager

    airtable_db = AirtableManager(base_id="app8vqaoWyOwa0sB5")
    collection = airtable_db.get_dataset_paths(
        collection_name=collection_name,
        version=collection_version,
    )

    print(f"  Total FOVs: {collection.total_fovs}")
    print(f"  Data paths: {[str(ds.data_path) for ds in collection.datasets]}")

print("\n✓ Full lineage chain:")
print(
    "  Airtable Collections → W&B Training Run → W&B Model Artifact → Checkpoint File"
)

# %%
# =============================================================================
# Part 5: Command Line Usage (Alternative to Python API)
# =============================================================================

print("\n=== CLI Usage Examples ===")

print(
    """
# Register a model after training:
python -m viscy.airtable.register_model \\
    logs/wandb/run-20260107/checkpoints/epoch=50.ckpt \\
    --name contrastive-rpe1 \\
    --type contrastive \\
    --version v2 \\
    --aliases production best \\
    --run-id 20260107-152420 \\
    --description "Best RPE1 model, val_loss=0.145"

# Checkpoints are copied to:
# /hpc/models/shared/contrastive/contrastive-rpe1-v2.ckpt

# View in W&B:
# https://wandb.ai/YOUR_ENTITY/viscy-model-registry/artifacts/model
"""
)

# %%
# =============================================================================
# Summary: When to use what
# =============================================================================

print("\n=== Summary ===")
print(
    """
Part 1 (Automatic - Already Implemented):
- CollectionWandbCallback logs collection metadata to every training run
- All experiments tracked automatically in W&B
- No manual work required

Part 2 (Manual - This Example):
- After training, manually register "blessed" models using register_model()
- Creates W&B artifact with lineage to training run
- Copies checkpoint to shared HPC directory
- Team can discover and load models via W&B UI or Python API

Workflow:
1. Train model with train_with_wandb.yml (automatic tracking)
2. Review metrics in W&B, identify best model
3. Register best checkpoint as artifact (manual)
4. Team can now load model by name/alias from registry
5. Full lineage: Collections → Training → Model Artifact

Benefits:
- Discoverability: Find models by collection, performance, date
- Versioning: v1, v2, v3 with aliases (production, latest)
- Lineage: Track which data/config produced which model
- Team collaboration: Shared registry + shared checkpoints
"""
)
