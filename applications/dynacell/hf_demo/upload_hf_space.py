"""Push hf_space/ to dihan-zheng/dynacell-demo on HuggingFace Spaces.

Run after upload_checkpoints.py:

    huggingface-cli login        # or set HF_TOKEN env var
    python upload_hf_space.py
"""

import os
from pathlib import Path
from huggingface_hub import HfApi

SPACE_REPO = "dihan-zheng/dynacell-demo"
HF_SPACE_DIR = Path(__file__).parent / "hf_space"


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    print(f"Creating / verifying Space: {SPACE_REPO}")
    api.create_repo(
        SPACE_REPO,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
    )

    print(f"Uploading {HF_SPACE_DIR} → {SPACE_REPO}")
    api.upload_folder(
        folder_path=str(HF_SPACE_DIR),
        repo_id=SPACE_REPO,
        repo_type="space",
    )
    print(f"\nDone.  Space URL: https://huggingface.co/spaces/{SPACE_REPO}")


if __name__ == "__main__":
    main()
