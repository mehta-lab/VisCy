"""Example usage of get_dataset_paths with Manifest and ManifestDataset dataclasses."""

# %%
from viscy.airtable.database import AirtableManager

BASE_ID = "app8vqaoWyOwa0sB5"
airtable_db = AirtableManager(base_id=BASE_ID)

# %%
# Fetch manifest from Airtable
manifest = airtable_db.get_dataset_paths(
    manifest_name="2024_11_07_A549_SEC61_DENV_wells_B1_B2",
    version="0.0.1",
)

# %%
# Manifest properties
print("=== Manifest ===")
print(f"manifest.name:       {manifest.name}")
print(f"manifest.version:    {manifest.version}")
print(f"len(manifest):       {len(manifest)} HCS plate(s)")
print(f"manifest.total_fovs: {manifest.total_fovs} FOVs")

# %%
# Iterate over ManifestDataset objects (one per HCS plate)
print("\n=== ManifestDataset ===")
for ds in manifest:
    print(f"ds.data_path:   {ds.data_path}")
    print(f"ds.tracks_path: {ds.tracks_path}")
    print(f"len(ds):        {len(ds)} FOVs")
    print(f"ds.fov_names:   {ds.fov_names[:3]}...")
    print(f"ds.fov_paths:   {ds.fov_paths[:2]}...")
    print(f"ds.exists():    {ds.exists()}")

# %%
# Validate paths exist (raises FileNotFoundError if not)
manifest.validate()
print("\nAll paths validated successfully!")


# %%
# List available manifests
print("=== Available Manifests ===")
df = airtable_db.list_manifests()
print(df[["name", "version", "purpose"]].dropna(subset=["name"]).to_string())

# %%
# =============================================================================
# Create TripletDataModule from manifest using factory function
# =============================================================================
from viscy.airtable.factory import create_triplet_datamodule_from_manifest

# Create data module from manifest
dm = create_triplet_datamodule_from_manifest(
    manifest=manifest,
    source_channel=["Phase3D"],
    z_range=(20, 21),
    batch_size=1,
    num_workers=1,
    initial_yx_patch_size=(160, 160),
    final_yx_patch_size=(160, 160),
    return_negative=False,
    time_interval=1,
)

# %%
# Setup and inspect the data module
dm.setup("fit")
print("\n=== TripletDataModule from Manifest ===")
print(f"Data module type: {type(dm).__name__}")
print(f"Train samples: {len(dm.train_dataset)}")
print(f"Val samples: {len(dm.val_dataset)}")

# %%
# =============================================================================
# Alternative: ManifestTripletDataModule (Lightning Config Compatible)
# =============================================================================
from viscy.airtable.factory import ManifestTripletDataModule

# This class is designed for Lightning CLI and config files
# but can also be used directly in Python
dm_class = ManifestTripletDataModule(
    base_id=BASE_ID,
    manifest_name="2024_11_07_A549_SEC61_DENV_wells_B1_B2",
    manifest_version="0.0.1",
    source_channel=["Phase3D"],
    z_range=(20, 21),
    batch_size=1,
    num_workers=1,
    initial_yx_patch_size=(160, 160),
    final_yx_patch_size=(160, 160),
    return_negative=False,
    time_interval=1,
)

dm_class.setup("fit")
print("\n=== ManifestTripletDataModule (Class) ===")
print(f"Data module type: {type(dm_class).__name__}")
print(f"Train samples: {len(dm_class.train_dataset)}")
print(f"Val samples: {len(dm_class.val_dataset)}")
print("Note: This class is designed for Lightning config files!")

# %% Visualize some of the images
import matplotlib.pyplot as plt
import torch

img_stack = []
for idx, batch in enumerate(dm.train_dataloader()):
    img_stack.append(batch["anchor"][0, 0, 0])
    if idx >= 9:
        break
img_stack = torch.stack(img_stack)
# %%
# Make subplot with 10 images
fig, axs = plt.subplots(2, 5, figsize=(10, 4))
for i in range(10):
    axs[i // 5, i % 5].imshow(img_stack[i], cmap="gray")
    axs[i // 5, i % 5].axis("off")
plt.show()
# %%
# =============================================================================
# Summary: When to use which approach
# =============================================================================
print("\n=== Summary ===")
print("Use create_triplet_datamodule_from_manifest() when:")
print("  - Working in Python scripts or notebooks")
print("  - Manifest has multiple HCS plates (auto-combines them)")
print("  - Already have manifest object loaded")
print("")
print("Use ManifestTripletDataModule when:")
print("  - Working with Lightning CLI and config files")
print("  - Training with single-plate manifests")
print("  - Want clean YAML configuration")
print("")
print("See examples/airtable/manifest_config_example.yml for config usage")
