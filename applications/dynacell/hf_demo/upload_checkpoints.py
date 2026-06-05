"""Upload all 12 dynacell checkpoints to mehta-lab/dynacell-checkpoints on HF Hub.

Run this once from the HPC where checkpoints are stored:

    pip install huggingface_hub
    huggingface-cli login        # or set HF_TOKEN env var
    python upload_checkpoints.py
"""

from pathlib import Path
from huggingface_hub import HfApi, create_repo

REPO_ID = "dihan-zheng/dynacell-checkpoints"

# (hf_filename, local_path)
CHECKPOINTS: list[tuple[str, str]] = [
    ("celldiff_caax.ckpt",
     "/hpc/projects/comp.micro/virtual_staining/models/cell_diff_vs_viscy"
     "/a549_mantis/memb/celldiff_r2/checkpoints/last.ckpt"),
    ("celldiff_h2b.ckpt",
     "/hpc/projects/comp.micro/virtual_staining/models/cell_diff_vs_viscy"
     "/a549_mantis/nucl/celldiff_r2/checkpoints/last.ckpt"),
    ("celldiff_sec61b.ckpt",
     "/hpc/projects/comp.micro/virtual_staining/models/cell_diff_vs_viscy"
     "/a549_mantis/sec61b/celldiff_r2/checkpoints/last.ckpt"),
    ("celldiff_tomm20.ckpt",
     "/hpc/projects/comp.micro/virtual_staining/models/cell_diff_vs_viscy"
     "/a549_mantis/tomm20/celldiff_r2/checkpoints/last.ckpt"),
    ("fnet3d_caax.ckpt",
     "/hpc/projects/comp.micro/virtual_staining/models/dynacell"
     "/a549_mantis/memb/fnet3d_paper/checkpoints/epoch=281-step=191760.ckpt"),
    ("fnet3d_h2b.ckpt",
     "/hpc/projects/comp.micro/virtual_staining/models/dynacell"
     "/a549_mantis/nucl/fnet3d_paper/checkpoints/epoch=293-step=199920.ckpt"),
    ("fnet3d_sec61b.ckpt",
     "/hpc/projects/comp.micro/virtual_staining/models/dynacell"
     "/a549_mantis/sec61b/fnet3d_paper/checkpoints/last.ckpt"),
    ("fnet3d_tomm20.ckpt",
     "/hpc/projects/comp.micro/virtual_staining/models/dynacell"
     "/a549_mantis/tomm20/fnet3d_paper/checkpoints/epoch=248-step=126990.ckpt"),
    ("vscyto3d_caax.ckpt",
     "/hpc/projects/comp.micro/virtual_staining/models/dynacell"
     "/a549_mantis/memb/fcmae_vscyto3d_pretrained/checkpoints/last.ckpt"),
    ("vscyto3d_h2b.ckpt",
     "/hpc/projects/comp.micro/virtual_staining/models/dynacell"
     "/a549_mantis/nucl/fcmae_vscyto3d_pretrained/checkpoints/last.ckpt"),
    ("vscyto3d_sec61b.ckpt",
     "/hpc/projects/comp.micro/virtual_staining/models/dynacell"
     "/a549_mantis/sec61b/fcmae_vscyto3d_pretrained_ws8500/checkpoints/last.ckpt"),
    ("vscyto3d_tomm20.ckpt",
     "/hpc/projects/comp.micro/virtual_staining/models/dynacell"
     "/a549_mantis/tomm20/fcmae_vscyto3d_pretrained_ws8500/checkpoints/last.ckpt"),
]


def main() -> None:
    import os
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    # Create repo if it doesn't exist yet
    create_repo(REPO_ID, repo_type="model", exist_ok=True, token=token)
    print(f"Repo: https://huggingface.co/{REPO_ID}")

    for hf_name, local_path in CHECKPOINTS:
        local = Path(local_path)
        if not local.exists():
            print(f"  SKIP (not found): {local_path}")
            continue
        print(f"  Uploading {hf_name}  ({local.stat().st_size / 1e9:.2f} GB) ...")
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=hf_name,
            repo_id=REPO_ID,
            repo_type="model",
        )
        print(f"  Done: {hf_name}")

    print("\nAll checkpoints uploaded.")


if __name__ == "__main__":
    main()
