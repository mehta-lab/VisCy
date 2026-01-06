"""Filter FOVs using pandas and create dataset tags."""

# %%
import os

from viscy.representation.airtable_fov_registry import AirtableFOVRegistry

BASE_ID = os.getenv("AIRTABLE_BASE_ID")
registry = AirtableFOVRegistry(base_id=BASE_ID)

# %%
# EXAMPLE 1: Get all FOVs as DataFrame and explore
print("=" * 70)
print("Getting all FOVs as DataFrame")
print("=" * 70)

df_fovs = registry.list_fovs()
print(f"\nTotal FOVs: {len(df_fovs)}")
print("\nDataFrame columns:")
print(df_fovs.columns.tolist())
print("\nFirst few rows:")
print(df_fovs.head())

# %%
# EXAMPLE 2: Filter by plate and rows B and C using pandas
print("\n" + "=" * 70)
print("Filter: Plate RPE1_plate1, Rows B and C, Good quality")
print("=" * 70)

# Get all FOVs as DataFrame
df = registry.list_fovs()

# Filter with pandas - simple and powerful!
filtered = df[
    (df["plate_name"] == "RPE1_plate1")
    & (df["quality"] == "Good")
    & (df["row"].isin(["B", "C"]))
]

print(f"\nTotal FOVs after filtering: {len(filtered)}")
print("\nBreakdown by well:")
print(filtered.groupby("well_id").size())

# Create dataset from filtered FOVs
fov_ids = filtered["fov_id"].tolist()

try:
    dataset_id = registry.create_dataset_from_fovs(
        dataset_name="RPE1_rows_BC_good",
        fov_ids=fov_ids,
        version="0.0.1",  # Semantic versioning
        purpose="training",
        description="Good quality FOVs from rows B and C",
    )
    print(f"\n✓ Created dataset: {dataset_id}")
    print(f"  Contains {len(fov_ids)} FOVs")
except ValueError as e:
    print(f"\n⚠ {e}")

# %%
# EXAMPLE 3: Group by plate and show summary
print("\n" + "=" * 70)
print("Group by plate and show summary")
print("=" * 70)

df_all = registry.list_fovs()

# Filter for good quality only
df_all = df_all[df_all["quality"] == "Good"]

grouped = df_all.groupby("plate_name")

for plate_name, group in grouped:
    print(f"\n{plate_name}:")
    print(f"  Total FOVs: {len(group)}")
    print(f"  Wells: {group['well_id'].unique()}")
    print(f"  Rows: {group['row'].unique()}")

# %%
# EXAMPLE 4: Complex filtering - specific rows and columns
print("\n" + "=" * 70)
print("Complex Filter: Rows B/C AND Columns 3/4")
print("=" * 70)

df = registry.list_fovs()

# Complex pandas filter: plate, quality, rows B or C, AND columns 3 or 4
filtered = df[
    (df["plate_name"] == "RPE1_plate1")
    & (df["quality"] == "Good")
    & (df["row"].isin(["B", "C"]))
    & (df["column"].isin(["3", "4"]))
]

print(f"\nFOVs matching criteria: {len(filtered)}")
print("\nBy well:")
print(filtered.groupby(["row", "column"]).size())

print("\nFOV IDs:")
for fov_id in filtered["fov_id"]:
    print(f"  {fov_id}")

# %%
# EXAMPLE 5: Exclude specific FOVs
print("\n" + "=" * 70)
print("Exclude specific FOVs from dataset")
print("=" * 70)

df = registry.list_fovs()

# Start with good quality FOVs from specific plate
filtered = df[(df["plate_name"] == "RPE1_plate1") & (df["quality"] == "Good")]

print(f"\nBefore exclusion: {len(filtered)} FOVs")

# List of FOVs to exclude (e.g., known contamination)
exclude_list = ["RPE1_plate1_B_3_2", "RPE1_plate1_C_4_1"]

# Filter out excluded FOVs
filtered = filtered[~filtered["fov_id"].isin(exclude_list)]

print(f"Excluded: {len(exclude_list)} FOVs")
print(f"After exclusion: {len(filtered)} FOVs")

# %%
# EXAMPLE 6: Summary statistics
print("\n" + "=" * 70)
print("Summary Statistics")
print("=" * 70)

df = registry.list_fovs()

print("\nFOVs per plate:")
print(df.groupby("plate_name").size())

print("\nFOVs per quality:")
print(df.groupby("quality").size())

print("\nFOVs per row (across all plates):")
print(df.groupby("row").size().sort_index())

print("\nWells with most FOVs:")
print(df.groupby("well_id").size().sort_values(ascending=False).head(10))
