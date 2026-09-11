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

try:
    from viscy_models.foundation import CellDinoModel, MorphEmModel
except ImportError:
    CellDinoModel = None  # type: ignore[assignment, misc]
    MorphEmModel = None  # type: ignore[assignment, misc]

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


def _require_cell_dino():
    if CellDinoModel is None:
        raise ImportError(
            "viscy_models.foundation.CellDinoModel is required for CellDinoFeatureExtractor. "
            "Install the in-tree workspace package via `uv sync --all-packages --all-extras` "
            "from the VisCy repo root, or `pip install -e packages/viscy-models`."
        )


def _require_morphem():
    if MorphEmModel is None:
        raise ImportError(
            "viscy_models.foundation.MorphEmModel is required for MorphEmFeatureExtractor. "
            "Install the in-tree workspace package via `uv sync --all-packages --all-extras` "
            "from the VisCy repo root, or `pip install -e packages/viscy-models`."
        )


class DynaCLRFeatureExtractor:
    """DynaCLR-based contrastive feature extractor for cell images.

    DynaCLR is trained on ``NormalizeSampled`` z-scored inputs (per-FOV or
    per-(FOV, timepoint) mean/std), so this extractor per-crop z-scores each
    2-D crop before the encoder to match the training input *shape*
    (zero-mean / unit-std). Per-crop statistics are used because the exact
    training-time per-FOV/timepoint stats are not available at eval for every
    store (e.g. the read-only A549 ``.ozx``); this leaves a scope difference
    (per masked crop vs whole-FOV) but closes the shape gap that raw [0, 1]
    crops left open.
    """

    # Version tag for the input-side preprocessing recipe. Stored alongside
    # cached features in the per-cache-dir manifest under
    # ``artifacts.dynaclr_features.<sha12>.preprocess_version``. Bump when
    # anything about how raw dataloader tensors are converted into model
    # input changes (normalization, channel handling, resize, etc.). The
    # cache layer compares cached vs current version on context init and
    # auto-invalidates the cached features for this extractor on mismatch.
    # The v2 bump folds two changes over the v1 raw-[0, 1] recipe: (1)
    # ``build_crops`` switched from raw min-max to robust percentile (1-99
    # clip + min-max) normalization upstream, and (2) per-crop spatial
    # z-score here, to match DynaCLR's ``NormalizeSampled`` z-scored training
    # distribution instead of feeding it bounded [0, 1] crops.
    PREPROCESS_VERSION = "v2"

    @staticmethod
    def _zscore(x: torch.Tensor) -> torch.Tensor:
        """Per-image spatial z-score over the trailing ``(H, W)`` dims.

        Uses the same ``(x - m) / (s + 1e-7)`` form (biased std) as
        :meth:`viscy_models.foundation.CellDinoModel.preprocess_2d` and the
        MorphEm wrapper, so all non-DINOv3 backbones share one z-score
        convention. Operates per leading index, so a batched ``(N, 1, 1, H, W)``
        tensor is normalized independently per crop.
        """
        m = x.mean(dim=(-2, -1), keepdim=True)
        s = x.std(dim=(-2, -1), unbiased=False, keepdim=True)
        return (x - m) / (s + 1e-7)

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
        image = self._zscore(image)
        with torch.inference_mode():
            features, _ = self.model(image)
        return features

    def extract_features_batch(self, images: list[np.ndarray], batch_size: int = 64) -> torch.Tensor:
        """Run the encoder over a batch of 2-D crops in one or more chunks.

        Stacks the crops to ``(N, 1, 1, H, W)`` so the contrastive encoder
        sees a real batch. Each crop is per-image z-scored (see
        :meth:`_zscore`) before the encoder. Chunks at ``batch_size`` to
        bound VRAM.
        """
        out_chunks: list[torch.Tensor] = []
        for i in range(0, len(images), batch_size):
            chunk = np.stack(images[i : i + batch_size], axis=0)
            batch = torch.as_tensor(chunk, device=self.model.device)[:, None, None, ...]
            batch = self._zscore(batch)
            with torch.inference_mode():
                features, _ = self.model(batch)
            out_chunks.append(features)
        return torch.cat(out_chunks, dim=0)


class DinoV3FeatureExtractor:
    """DINOv3-based feature extractor for cell images."""

    # Version tag for the input-side preprocessing recipe. Stored under
    # ``artifacts.dinov3_features.<slug>.preprocess_version`` in the cache
    # manifest. Current recipe is per-image robust percentile-clip
    # (``[1, 99]``) + min-max scale to ``[0, 1]`` (done upstream by
    # ``build_crops``) followed by ImageNet ``(mean, std)`` normalization
    # via ``AutoImageProcessor`` with ``do_rescale=False`` — the processor's
    # default ``rescale_factor`` of ``1/255`` would otherwise divide our
    # [0, 1] float crops a second time, leaving the model with
    # essentially-black inputs and features cosine-uncorrelated with the
    # intended representation. DINOv3 has no internal per-image z-score, so
    # this upstream scaling is load-bearing: the v3 bump invalidates every
    # v2 cache entry, which used raw min-max crops where a single hot pixel
    # in the FOV compressed all crops toward black and injected a GT-vs-pred
    # intensity-range mismatch. v2 in turn had fixed the v1 double-rescale.
    PREPROCESS_VERSION = "imagenet_normalize_v3"

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
        # Belt-and-suspenders: HF defaults `do_rescale=True` with
        # `rescale_factor=1/255` (uint8 → [0, 1]) on the processor instance.
        # Our crops are already float [0, 1] from ``build_crops`` (robust
        # percentile normalization), so the rescale must not run. Per-call
        # ``do_rescale=False`` below covers
        # the current invocations; pinning the instance attribute here
        # keeps any future helper that calls ``self.processor(...)`` without
        # the kwarg from silently regressing to the double-rescaled path.
        self.processor.do_rescale = False
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
        inputs = self.processor(images=image, return_tensors="pt", do_rescale=False).to(self.model.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        return outputs.pooler_output

    def extract_features_batch(self, images: list[np.ndarray], batch_size: int = 32) -> torch.Tensor:
        """Run the ViT backbone over a batch of 2-D crops in one or more chunks.

        AutoImageProcessor accepts a list of 3-channel images and produces
        a stacked tensor; we chunk at ``batch_size`` to bound VRAM.
        """
        out_chunks: list[torch.Tensor] = []
        for i in range(0, len(images), batch_size):
            chunk = [np.stack([img] * 3, axis=0) for img in images[i : i + batch_size]]
            inputs = self.processor(images=chunk, return_tensors="pt", do_rescale=False).to(self.model.device)
            with torch.inference_mode():
                outputs = self.model(**inputs)
            out_chunks.append(outputs.pooler_output)
        return torch.cat(out_chunks, dim=0)


class CellDinoFeatureExtractor:
    """CELL-DINO foundation model (DINOv2 ViT-L/16 pretrained on HPA) for cell images.

    Wraps :class:`viscy_models.foundation.CellDinoModel` so the eval pipeline
    can use it via the same ``extract_features(image_2d)`` contract as the
    DINOv3 and DynaCLR extractors. The underlying model's ``preprocess_2d``
    handles the 224×224 resize and per-image per-channel spatial z-score
    (the ``self_normalize`` recipe used during CELL-DINO pretraining), so
    the caller can feed the same masked 2-D cell crop used for the other
    backbones.
    """

    # Version tag for the input-side preprocessing recipe. Stored under
    # ``artifacts.celldino_features.<sha12>.preprocess_version`` in the
    # cache manifest. Current recipe is per-image per-channel spatial
    # z-score (the official ``self_normalize`` recipe used during
    # CELL-DINO pretraining), applied in
    # :meth:`viscy_models.foundation.CellDinoModel.preprocess_2d`. Bump on
    # any future change so cached features auto-invalidate. The bump from
    # the previous min/max-to-[0,1] recipe (which had no version tag) is
    # one such transition — see commit e648c4ce. The v2 bump reflects
    # ``build_crops`` switching to robust percentile (1-99 clip + min-max)
    # normalization upstream: the percentile clip is not affine, so it
    # changes the z-scored input even though the min-max prescale alone
    # would cancel under the per-image z-score.
    PREPROCESS_VERSION = "self_normalize_v2"

    def __init__(self, weights_path: str, img_size: int = 224, patch_size: int = 16):
        """Load a CELL-DINO checkpoint from a local ``.pth`` state_dict.

        Parameters
        ----------
        weights_path :
            Absolute path to the CELL-DINO ``.pth`` state_dict (e.g. the
            ``channel_adaptive_dino_vitl16_pretrain_cells-*.pth`` published
            at ``/hpc/projects/organelle_phenotyping/models/CELL-DINO/``).
        img_size :
            Spatial size the model interpolates inputs to, by default 224.
        patch_size :
            ViT patch size, by default 16.
        """
        _require_cell_dino()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CellDinoModel(weights_path=weights_path, img_size=img_size, patch_size=patch_size, freeze=True)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def extract_features(self, image: np.ndarray) -> torch.Tensor:
        """Extract the channel-mean cls token from a 2-D image patch.

        Parameters
        ----------
        image :
            2-D array (H, W); wrapped to ``(1, 1, H, W)`` so CellDinoModel
            treats it as a single-channel, single-batch input.

        Returns
        -------
        torch.Tensor
            1-D embedding vector of shape ``(1024,)``.
        """
        x = torch.as_tensor(image, device=self.device, dtype=torch.float32)[None, None, ...]
        with torch.inference_mode():
            features, _ = self.model(x)
        return features

    def extract_features_batch(self, images: list[np.ndarray], batch_size: int = 32) -> torch.Tensor:
        """Run CELL-DINO over a batch of 2-D crops in one or more chunks.

        Stacks the crops to ``(N, 1, H, W)`` so the ViT-L/16 backbone runs
        once per chunk. Chunks at ``batch_size`` to bound VRAM — at
        ``img_size=224, patch_size=16``, ViT-L activations are ~1 GB per
        32 cells in fp32.
        """
        out_chunks: list[torch.Tensor] = []
        for i in range(0, len(images), batch_size):
            chunk = np.stack(images[i : i + batch_size], axis=0)
            batch = torch.as_tensor(chunk, device=self.device, dtype=torch.float32)[:, None, ...]
            with torch.inference_mode():
                features, _ = self.model(batch)
            out_chunks.append(features)
        return torch.cat(out_chunks, dim=0)


class MorphEmFeatureExtractor:
    """MorphEm (CaicedoLab/MorphEm, DINO ViT-S/16 on microscopy) embedder for cell images.

    Wraps :class:`viscy_models.foundation.MorphEmModel` so the eval pipeline
    can use it via the same ``extract_features(image_2d)`` contract as the
    DINOv3 / DynaCLR / CELL-DINO extractors. Unlike :class:`CellDinoModel`,
    ``MorphEmModel.forward`` does **not** normalize inline, so this extractor
    calls :meth:`viscy_models.foundation.MorphEmModel.preprocess_2d`
    explicitly (per-image z-score then resize to 224) before the backbone.
    """

    # Version tag for the input-side preprocessing recipe. Stored under
    # ``artifacts.morphem_features.<slug>.preprocess_version`` in the cache
    # manifest. Current recipe is per-image per-channel spatial z-score
    # (PerImageNormalize) then bilinear resize to 224, applied in
    # :meth:`viscy_models.foundation.MorphEmModel.preprocess_2d`. Bump on any
    # future change so cached features auto-invalidate. The v2 bump reflects
    # ``build_crops`` switching to robust percentile (1-99 clip + min-max)
    # normalization upstream: the percentile clip is not affine, so it
    # changes the z-scored input even though the min-max prescale alone
    # would cancel under the per-image z-score.
    PREPROCESS_VERSION = "per_image_norm_v2"

    def __init__(self, pretrained_model_name: str, revision: str | None = None, img_size: int = 224):
        """Load MorphEm from a HuggingFace hub id (or local snapshot dir).

        Parameters
        ----------
        pretrained_model_name :
            HuggingFace id (``"CaicedoLab/MorphEm"``) or a local snapshot
            directory; resolved from the shared ``HF_HUB_CACHE``.
        revision :
            Hub commit SHA to pin the ``trust_remote_code`` model to. The
            eval config sets it; ``None`` (hub head) is only for ad-hoc use.
        img_size :
            Spatial size the model interpolates inputs to, by default 224.
        """
        _require_morphem()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MorphEmModel(model_name=pretrained_model_name, revision=revision, img_size=img_size, freeze=True)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def extract_features(self, image: np.ndarray) -> torch.Tensor:
        """Extract the channel-mean CLS token from a 2-D image patch.

        Parameters
        ----------
        image :
            2-D array (H, W); wrapped to ``(1, 1, H, W)`` so MorphEm treats
            it as a single-channel, single-batch input.

        Returns
        -------
        torch.Tensor
            Batch embedding of shape ``(1, D)`` (one row; ``D`` = 384 for the
            ViT-S/16 backbone), matching the ``(N, D)`` extractor contract.
        """
        x = torch.as_tensor(image, device=self.device, dtype=torch.float32)[None, None, ...]
        with torch.inference_mode():
            x = self.model.preprocess_2d(x)
            features, _ = self.model(x)
        return features

    def extract_features_batch(self, images: list[np.ndarray], batch_size: int = 32) -> torch.Tensor:
        """Run MorphEm over a batch of 2-D crops in one or more chunks.

        Stacks the crops to ``(N, 1, H, W)`` and chunks at ``batch_size`` to
        bound VRAM; each chunk is per-image z-scored + resized before the
        ViT-S/16 backbone runs.
        """
        out_chunks: list[torch.Tensor] = []
        for i in range(0, len(images), batch_size):
            chunk = np.stack(images[i : i + batch_size], axis=0)
            batch = torch.as_tensor(chunk, device=self.device, dtype=torch.float32)[:, None, ...]
            with torch.inference_mode():
                batch = self.model.preprocess_2d(batch)
                features, _ = self.model(batch)
            out_chunks.append(features)
        return torch.cat(out_chunks, dim=0)


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

    # Group / sort once: with ~100 feature-metric columns the per-column
    # re-groupby and per-(column, FOV) boolean mask dominated this function.
    by_fov = df.groupby("FOV")
    all_fov_means = by_fov[metric_cols].mean()
    multi_tp_fovs = by_fov["Timepoint"].nunique().pipe(lambda s: s[s > 1].index.tolist())
    tp_frames = {fov: by_fov.get_group(fov).sort_values("Timepoint") for fov in multi_tp_fovs}

    for col in metric_cols:
        # --- Plot 1: mean per FOV ---
        fov_means = all_fov_means[col]
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
            for fov, fov_df in tp_frames.items():
                ax.plot(fov_df["Timepoint"], fov_df[col], marker="o", label=fov)
            ax.set_xlabel("Timepoint")
            ax.set_ylabel(col)
            ax.set_title(f"{col} — per Timepoint (multi-timepoint FOVs)")
            ax.legend(fontsize=7, loc="best")
            fig.tight_layout()
            fig.savefig(plot_dir / f"{col}_timepoints.png", dpi=150)
            plt.close(fig)
