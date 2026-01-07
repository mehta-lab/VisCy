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
