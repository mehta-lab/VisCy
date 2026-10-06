"""Upload the demo-data zips and card to biohub/dynacell-demo-data on HF Hub.

The dataset repo is private and lives in the biohub "Dynacell" resource group (see
AGENT.md). Build the zips first with build_demo_data.py, then:

    hf auth login                # or set HF_TOKEN env var
    python upload_demo_data.py ZIP_DIR
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "biohub/dynacell-demo-data"
MARKERS = ("CAAX", "H2B", "SEC61B", "TOMM20")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("zip_dir", type=Path)
    args = ap.parse_args()
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    for marker in MARKERS:
        name = f"{marker}_mock.zarr.zip"
        path = args.zip_dir / name
        print(f"  Uploading {name}  ({path.stat().st_size / 1e6:.0f} MB) ...")
        api.upload_file(path_or_fileobj=str(path), path_in_repo=name, repo_id=REPO_ID, repo_type="dataset")
    api.upload_file(
        path_or_fileobj=str(Path(__file__).parent / "cards" / "demo_data_README.md"),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print(f"Done: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
