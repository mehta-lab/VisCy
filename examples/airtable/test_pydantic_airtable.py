"""Test Pydantic dataset registration with Airtable."""

from viscy.airtable import AirtableManager, DatasetRecord

BASE_ID = "app8vqaoWyOwa0sB5"

print("=" * 70)
print("Testing Pydantic + Airtable Integration")
print("=" * 70)

airtable_db = AirtableManager(base_id=BASE_ID)

# %%
# Test 1: Single dataset registration
print("\n[Test 1] Register single dataset")
print("-" * 70)

dataset = DatasetRecord(
    dataset_name="pydantic_test_plate",
    well_id="A_1",
    fov_name="0",
    data_path="/hpc/data/pydantic_test.zarr/A/1/0",
    cell_type="A549",
    organelle="SEC61B",
)

try:
    record_id = airtable_db.register_dataset(dataset)
    print(f"✓ Registered: {record_id}")
    print(f"  Dataset: {dataset.dataset_name}")
    print(f"  Well: {dataset.well_id}, FOV: {dataset.fov_name}")
    print(f"  Path: {dataset.data_path}")
except Exception as e:
    print(f"✗ Failed: {e}")

# %%
# Test 2: Bulk dataset registration
print("\n[Test 2] Register multiple datasets")
print("-" * 70)

datasets = [
    DatasetRecord(
        dataset_name="pydantic_test_plate",
        well_id=f"B_{well}",
        fov_name=str(fov),
        data_path=f"/hpc/data/pydantic_test.zarr/B/{well}/{fov}",
        cell_type="A549",
    )
    for well in range(1, 3)
    for fov in range(2)
]

try:
    record_ids = airtable_db.register_datasets(datasets)
    print(f"✓ Registered {len(record_ids)} datasets")
    for ds, rec_id in zip(datasets, record_ids):
        print(f"  {ds.dataset_name}_{ds.well_id}_{ds.fov_name} -> {rec_id}")
except Exception as e:
    print(f"✗ Failed: {e}")

# %%
# Test 3: List datasets as Pydantic models
print("\n[Test 3] List datasets as Pydantic models")
print("-" * 70)

try:
    all_datasets = airtable_db.list_datasets(as_pydantic=True)
    print(f"Total datasets in Airtable: {len(all_datasets)}")

    # Filter for our test datasets
    test_datasets = [ds for ds in all_datasets if ds.fov_id.startswith("pydantic_test")]
    print(f"\nTest datasets found: {len(test_datasets)}")
    for ds in test_datasets:
        print(f"  - {ds.fov_id}")
        print(f"    Dataset: {ds.dataset_name}")
        print(f"    Cell type: {ds.cell_type}")
        print(f"    Record ID: {ds.record_id}")

except Exception as e:
    print(f"✗ Failed: {e}")

print("\n" + "=" * 70)
print("Airtable integration test complete!")
print("=" * 70)
