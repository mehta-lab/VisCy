"""Simple example: Register a single dataset to Airtable."""

from viscy.airtable import AirtableManager, DatasetRecord

BASE_ID = "app8vqaoWyOwa0sB5"

# Create dataset with validation
dataset = DatasetRecord(
    dataset_name="2024_11_07_A549_SEC61_DENV",
    well_id="B/1",
    fov_name="0",
    data_path="/hpc/data/2024_11_07_A549_SEC61_DENV.zarr/B/1/0",
    cell_type="A549",
    organelle="SEC61B",
    channel_0="brightfield",
    channel_1="nucleus",
    channel_2="protein",
)

print("Dataset to register:")
print(f"  {dataset}")
print(f"  Dataset: {dataset.dataset_name}")
print(f"  Well: {dataset.well_id}, FOV: {dataset.fov_name}")
print(f"  Cell type: {dataset.cell_type}")
print(f"  Organelle: {dataset.organelle}")
print(f"  Path: {dataset.data_path}")

# Register to Airtable
print("\nRegistering to Airtable...")
airtable_db = AirtableManager(base_id=BASE_ID)
record_id = airtable_db.register_dataset(dataset)

print("\n✓ Successfully registered!")
print(f"  Airtable Record ID: {record_id}")
print(
    f"  FOV_ID will be auto-generated as: {dataset.dataset_name}_{dataset.well_id}_{dataset.fov_name}"
)
