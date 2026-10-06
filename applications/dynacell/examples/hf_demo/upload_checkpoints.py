"""Upload the 12 released A549 dynacell checkpoints to biohub/dynacell-checkpoints on HF Hub.

The repo is private and lives in the biohub "Dynacell" resource group (see
AGENT.md). Run this from the HPC where checkpoints are stored:

    pip install huggingface_hub
    hf auth login                # or set HF_TOKEN env var
    python upload_checkpoints.py
"""

from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ID = "biohub/dynacell-checkpoints"
# biohub "Dynacell" resource group (see AGENT.md).
RESOURCE_GROUP_ID = "6a234bb4507cbbbb04456767"

# Release zoo root (mirrors s3://dynacell/v1/models/); see models/checkpoints.csv.
RELEASE_A549 = "/hpc/projects/virtual_staining/dynacell_v1/models/a549"

# (hf_filename, local_path) -- the released A549-trained checkpoints.
CHECKPOINTS: list[tuple[str, str]] = [
    ("celldiff_caax.ckpt", f"{RELEASE_A549}/membrane/celldiff/epoch=19-step=86400.ckpt"),
    ("celldiff_h2b.ckpt", f"{RELEASE_A549}/nucleus/celldiff/epoch=19-step=86400.ckpt"),
    ("celldiff_sec61b.ckpt", f"{RELEASE_A549}/er/celldiff/epoch=19-step=66240-v1.ckpt"),
    ("celldiff_tomm20.ckpt", f"{RELEASE_A549}/mito/celldiff/epoch=19-step=64800-v1.ckpt"),
    ("fnet3d_caax.ckpt", f"{RELEASE_A549}/membrane/fnet3d/epoch=281-step=191760.ckpt"),
    ("fnet3d_h2b.ckpt", f"{RELEASE_A549}/nucleus/fnet3d/epoch=293-step=199920.ckpt"),
    ("fnet3d_sec61b.ckpt", f"{RELEASE_A549}/er/fnet3d/epoch=350-step=185679.ckpt"),
    ("fnet3d_tomm20.ckpt", f"{RELEASE_A549}/mito/fnet3d/epoch=170-step=87210.ckpt"),
    ("vscyto3d_caax.ckpt", f"{RELEASE_A549}/membrane/vscyto3d/epoch=121-step=26474.ckpt"),
    ("vscyto3d_h2b.ckpt", f"{RELEASE_A549}/nucleus/vscyto3d/epoch=134-step=29295.ckpt"),
    ("vscyto3d_sec61b.ckpt", f"{RELEASE_A549}/er/vscyto3d/epoch=106-step=18404.ckpt"),
    ("vscyto3d_tomm20.ckpt", f"{RELEASE_A549}/mito/vscyto3d/epoch=115-step=18328.ckpt"),
]


def main() -> None:
    import os

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    # Create repo if it doesn't exist yet (private, in the Dynacell resource group)
    create_repo(
        REPO_ID,
        repo_type="model",
        private=True,
        resource_group_id=RESOURCE_GROUP_ID,
        exist_ok=True,
        token=token,
    )
    print(f"Repo: https://huggingface.co/{REPO_ID}")

    for hf_name, local_path in CHECKPOINTS:
        local = Path(local_path)
        print(f"  Uploading {hf_name}  ({local.stat().st_size / 1e9:.2f} GB) ...")
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=hf_name,
            repo_id=REPO_ID,
            repo_type="model",
        )
        print(f"  Done: {hf_name}")

    api.upload_file(
        path_or_fileobj=str(Path(__file__).parent / "cards" / "checkpoints_README.md"),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model",
    )
    print("\nAll checkpoints and the model card uploaded.")


if __name__ == "__main__":
    main()
