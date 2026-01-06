"""Filter datasets using pandas and create manifest tags."""

# %%

from viscy.airtable.datasets import AirtableDatasets

# BASE_ID = os.getenv("AIRTABLE_BASE_ID")
BASE_ID = "app8vqaoWyOwa0sB5"
registry = AirtableDatasets(base_id=BASE_ID)

# %%
# EXAMPLE 1: Get all dataset records as DataFrame and explore
print("=" * 70)
print("Getting all dataset records as DataFrame")
print("=" * 70)

df_datasets = registry.list_datasets()
print(f"\nTotal dataset records: {len(df_datasets)}")
print("\nDataFrame columns:")
print(df_datasets.columns.tolist())
print("\nFirst few rows:")
print(df_datasets.head())

# %%
# EXAMPLE 2: Filter by dataset and specific wells using pandas
print("\n" + "=" * 70)
print("Filter: Dataset, Wells B_3 and B_4")
print("=" * 70)

# Get all dataset records as DataFrame
df = registry.list_datasets()

# Filter with pandas - simple and powerful!
filtered = df[
    (df["Dataset"] == "2024_11_07_A549_SEC61_DENV")
    & (df["Well ID"].isin(["B/1", "B/2"]))
]

print(f"\nTotal dataset records after filtering: {len(filtered)}")
print("\nBreakdown by well:")
print(filtered.groupby("Well ID").size())

# Create manifest from filtered dataset records
fov_ids = filtered["FOV_ID"].tolist()

try:
    manifest_id = registry.create_manifest_from_datasets(
        manifest_name="2024_11_07_A549_SEC61_DENV_wells_B1_B2",
        fov_ids=fov_ids,
        version="0.0.1",  # Semantic versioning
        purpose="training",
        description="Dataset records from wells B_3 and B_4",
    )
    print(f"\n✓ Created manifest: {manifest_id}")
    print(f"  Contains {len(fov_ids)} dataset records")
except ValueError as e:
    print(f"\n⚠ {e}")

# %%
# Delete the manifest entry demo
registry.delete_manifest(manifest_id)
print(f"Deleted manifest: {manifest_id}")

# %%
# EXAMPLE 3: Group by dataset and show summary
print("\n" + "=" * 70)
print("Group by dataset and show summary")
print("=" * 70)

df_all = registry.list_datasets()

grouped = df_all.groupby("Dataset")

for dataset_name, group in grouped:
    print(f"\n{dataset_name}:")
    print(f"  Total records: {len(group)}")
    print(f"  Wells: {sorted(group['Well ID'].unique())}")

# %%
# EXAMPLE 4: Filter by multiple wells
print("\n" + "=" * 70)
print("Filter: Multiple specific wells")
print("=" * 70)

df = registry.list_datasets()

# Filter for specific wells from a dataset
filtered = df[
    (df["Dataset"] == "2024_11_07_A549_SEC61_DENV")
    & (df["Well ID"].isin(["B/3", "B/4", "C/3", "C/4"]))
]

print(f"\nDataset records matching criteria: {len(filtered)}")
print("\nBy well:")
print(filtered.groupby("Well ID").size())

print("\nFOV IDs:")
for fov_id in filtered["FOV_ID"]:
    print(f"  {fov_id}")

# %%
# EXAMPLE 5: Summary statistics
print("\n" + "=" * 70)
print("Summary Statistics")
print("=" * 70)

df = registry.list_datasets()

print("\nDataset records per source dataset:")
print(df.groupby("Dataset").size())

print("\nWells with most dataset records:")
print(df.groupby("Well ID").size().sort_values(ascending=False).head(10))

print("\nTotal unique wells:")
print(f"{df['Well ID'].nunique()} wells")

print("\nTotal unique FOV IDs:")
print(f"{df['FOV_ID'].nunique()} FOV IDs")

# %%
