# ruff: noqa: I001 — matplotlib.use() must be called before pyplot import
"""Feature extraction utilities and plotting helpers for evaluation."""

import numpy as np
import torch
import matplotlib

try:
    from transformers import AutoModel, AutoImageProcessor
except ImportError:
    AutoModel = None  # type: ignore[assignment, misc]
    AutoImageProcessor = None  # type: ignore[assignment, misc]

try:
    from dynaclr.engine import ContrastiveModule
except ImportError:
    ContrastiveModule = None  # type: ignore[assignment, misc]

try:
    from viscy_models.contrastive import ContrastiveEncoder
except ImportError:
    ContrastiveEncoder = None  # type: ignore[assignment, misc]

matplotlib.use("Agg")
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def _require_transformers():
    if AutoModel is None:
        raise ImportError(
            "transformers is required for DinoV3FeatureExtractor. Install it with: pip install transformers"
        )


def _require_dynaclr():
    if ContrastiveModule is None:
        raise ImportError("dynaclr is required for DynaCLRFeatureExtractor. Install it with: pip install dynaclr")


def _require_viscy_models():
    if ContrastiveEncoder is None:
        raise ImportError(
            "viscy_models is required for DynaCLRFeatureExtractor. Install it with: pip install viscy-models"
        )


class DynaCLRFeatureExtractor:
    """DynaCLR-based contrastive feature extractor for cell images."""

    def __init__(self, checkpoint: str, encoder_config: dict):
        """Load DynaCLR model from checkpoint.

        Parameters
        ----------
        checkpoint :
            Path to a Lightning checkpoint file.
        encoder_config :
            Keyword arguments for ``ContrastiveEncoder`` (backbone, channels, etc.).
        """
        _require_dynaclr()
        _require_viscy_models()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = ContrastiveEncoder(**encoder_config)
        self.model = ContrastiveModule.load_from_checkpoint(checkpoint, map_location="cpu", encoder=encoder)
        self.model.to(device)
        self.model.eval()

    def extract_features(self, image: np.ndarray) -> torch.Tensor:
        """Extract embedding from a 2-D image patch.

        Parameters
        ----------
        image :
            2-D array (H, W); will be wrapped to (1, 1, 1, H, W).

        Returns
        -------
        torch.Tensor
            1-D embedding vector of shape ``(embedding_dim,)``.
        """
        image = torch.as_tensor(image, device=self.model.device)[None, None, None, ...]
        with torch.inference_mode():
            features, _ = self.model(image)
        return features


class DinoV3FeatureExtractor:
    """DINOv3-based feature extractor for cell images."""

    def __init__(self, pretrained_model_name: str):
        """Load DINOv3 model from HuggingFace Hub.

        Parameters
        ----------
        pretrained_model_name :
            HuggingFace model identifier, e.g.
            ``"facebook/dinov3-convnext-base-pretrain-lvd1689m"``.
        """
        _require_transformers()
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
        self.model = AutoModel.from_pretrained(
            pretrained_model_name,
            device_map="auto",
        )

    def extract_features(self, image: np.ndarray) -> torch.Tensor:
        """Extract pooled features from a 2-D image patch.

        Parameters
        ----------
        image :
            2-D array (H, W); replicated to 3 channels for the ViT backbone.

        Returns
        -------
        torch.Tensor
            Pooled output tensor.
        """
        # Replicate single channel to 3 channels expected by the ViT backbone
        image = np.stack([image] * 3, axis=0)
        inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        return outputs.pooler_output


def _minmax_norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Min-max normalize array to [0, 1]."""
    return (x - x.min()) / (x.max() - x.min() + eps)


def plot_metrics(df: pd.DataFrame, save_dir: Path, metric_type: str) -> None:
    """Plot metrics per FOV and, when applicable, over time.

    For each metric column (every column except ``FOV`` and ``Timepoint``):

    1. **Mean-per-FOV bar chart** -- y-axis is the value averaged over all
       Timepoints for each FOV; x-axis is the FOV name.  Saved to
       ``save_dir / metric_type / <metric>_fov_mean.png``.

    2. **Timepoint line chart** -- only produced when at least one FOV has more
       than one Timepoint.  Each such FOV is drawn as a separate line.  Saved
       to ``save_dir / metric_type / <metric>_timepoints.png``.

    Parameters
    ----------
    df :
        DataFrame with at least ``FOV`` and ``Timepoint`` columns plus one or
        more metric columns.
    save_dir :
        Root results directory.
    metric_type :
        Subfolder name, e.g. ``"pixel_metrics"``, ``"mask_metrics"``, or
        ``"feature_metrics"``.
    """
    plot_dir = save_dir / metric_type
    plot_dir.mkdir(parents=True, exist_ok=True)

    metric_cols = [c for c in df.columns if c not in ("FOV", "Timepoint")]

    # FOVs with more than one timepoint
    multi_tp_fovs = df.groupby("FOV")["Timepoint"].nunique().pipe(lambda s: s[s > 1].index.tolist())

    for col in metric_cols:
        # --- Plot 1: mean per FOV ---
        fov_means = df.groupby("FOV")[col].mean()
        n_fovs = len(fov_means)

        fig, ax = plt.subplots(figsize=(max(6, n_fovs * 0.7), 5))
        ax.bar(range(n_fovs), fov_means.values)
        ax.set_xticks(range(n_fovs))
        ax.set_xticklabels(fov_means.index, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("FOV")
        ax.set_ylabel(col)
        ax.set_title(f"{col} — mean per FOV")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{col}_fov_mean.png", dpi=150)
        plt.close(fig)

        # --- Plot 2: metric over Timepoint for multi-timepoint FOVs ---
        if multi_tp_fovs:
            fig, ax = plt.subplots(figsize=(8, 5))
            for fov in multi_tp_fovs:
                fov_df = df[df["FOV"] == fov].sort_values("Timepoint")
                ax.plot(fov_df["Timepoint"], fov_df[col], marker="o", label=fov)
            ax.set_xlabel("Timepoint")
            ax.set_ylabel(col)
            ax.set_title(f"{col} — per Timepoint (multi-timepoint FOVs)")
            ax.legend(fontsize=7, loc="best")
            fig.tight_layout()
            fig.savefig(plot_dir / f"{col}_timepoints.png", dpi=150)
            plt.close(fig)
