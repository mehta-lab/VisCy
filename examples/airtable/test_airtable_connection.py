#!/usr/bin/env python3
"""Test Airtable connection and setup."""

# %%
import os

from pyairtable import Api

print("=" * 70)
print("Testing Airtable Connection")
print("=" * 70)

# Check environment variables
# TODO: Add these ENVIRONMENT VARIABLES TO BASHRC or export them in the node
api_key = os.getenv("AIRTABLE_API_KEY")
base_id = os.getenv("AIRTABLE_BASE_ID")

print("\n1. Environment Variables:")
print(
    f"   AIRTABLE_API_KEY: {'✓ Set' if api_key else '✗ Not set'} ({api_key[:10] if api_key else 'N/A'}...)"
)
print(
    f"   AIRTABLE_BASE_ID: {'✓ Set' if base_id else '✗ Not set'} ({base_id if base_id else 'N/A'})"
)

if not api_key or not base_id:
    print("\n❌ ERROR: Environment variables not set!")
    print("\nRun these commands in your shell:")
    print('  export AIRTABLE_API_KEY="patXXXXXXXXXXXXXX"')
    print('  export AIRTABLE_BASE_ID="appXXXXXXXXXXXXXX"')
    print("\nOr add them to your ~/.bashrc")
    exit(1)

# Test API connection
print("\n2. Testing API Connection...")
try:
    api = Api(api_key)
    print("   ✓ API initialized")
except Exception as e:
    print(f"   ✗ Failed to initialize API: {e}")
    exit(1)

# Test Datasets table
print("\n3. Testing Datasets Table...")
try:
    datasets_table = api.table(base_id, "Datasets")
    records = datasets_table.all()
    print("   ✓ Connected to Datasets table")
    print(f"   ✓ Found {len(records)} record(s)")

    if records:
        print("\n   Existing datasets:")
        for record in records:
            fields = record["fields"]
            name = fields.get("name", "N/A")
            version = fields.get("version", "N/A")
            print(f"   - {name} (v{version})")
    else:
        print("   (Table is empty - this is OK for first run)")

except Exception as e:
    print(f"   ✗ Failed to access Datasets table: {e}")
    print("\n   Make sure you created a table named 'Datasets' (case-sensitive)")
    exit(1)

# Test Models table
print("\n4. Testing Models Table...")
try:
    models_table = api.table(base_id, "Models")
    records = models_table.all()
    print("   ✓ Connected to Models table")
    print(f"   ✓ Found {len(records)} record(s)")

    if records:
        print("\n   Existing models:")
        for record in records:
            fields = record["fields"]
            name = fields.get("model_name", "N/A")
            acc = fields.get("test_accuracy", "N/A")
            print(f"   - {name} (accuracy: {acc})")
    else:
        print("   (Table is empty - this is OK for first run)")

except Exception as e:
    print(f"   ✗ Failed to access Models table: {e}")
    print("\n   Make sure you created a table named 'Models' (case-sensitive)")
    exit(1)

# Test creating a dummy record
print("\n5. Testing Write Permissions...")
try:
    test_record = datasets_table.create(
        {
            "name": "connection_test",
            "version": "v0",
            "hpc_path": "/tmp/test",
            "sha256": "test_hash",
            "created_date": "2024-12-19T00:00:00",
        }
    )
    print(f"   ✓ Successfully created test record (ID: {test_record['id']})")

    # Clean up
    datasets_table.delete(test_record["id"])
    print("   ✓ Successfully deleted test record")

except Exception as e:
    print(f"   ✗ Failed to write to Datasets table: {e}")
    print("\n   Check your API token has 'data.records:write' scope")
    exit(1)

print("\n" + "=" * 70)
print("✅ SUCCESS: Airtable is configured correctly!")
# %%
