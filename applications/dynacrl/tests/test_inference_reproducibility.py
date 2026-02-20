"""Integration tests for inference reproducibility of modular DynaCLR.

Validates that the modular ContrastiveModule produces identical inference
results to the original monolithic VisCy code. Tests checkpoint loading,
embedding prediction, and numerical exactness.

Tolerance rationale: GPU convolution non-determinism across CUDA/cuDNN
versions and hardware causes small numerical differences in deep ConvNeXt
models. Observed statistics (A40 GPU, CUDA 12.x):
  - Mean absolute diff: ~0.0006
  - 99.9th percentile: ~0.004
  - Max absolute diff: ~0.02
  - Pearson correlation: >0.999 (features), >0.99999 (projections)
We use atol=0.02 to accommodate cross-environment GPU non-determinism
while rejecting any functional divergence.

Requirements: INFER-01, INFER-02, INFER-03, TEST-01, TEST-02
"""

import numpy as np
import pytest
import torch
from dynaclr.engine import ContrastiveModule
from lightning.pytorch import Trainer, seed_everything
from scipy import stats

from viscy_models.contrastive import ContrastiveEncoder
from viscy_transforms import NormalizeSampled

from .conftest import requires_hpc_and_gpu

ENCODER_KWARGS = {
    "backbone": "convnext_tiny",
    "in_channels": 1,
    "in_stack_depth": 1,
    "stem_kernel_size": [1, 4, 4],
    "stem_stride": [1, 4, 4],
    "embedding_dim": 768,
    "projection_dim": 32,
    "drop_path_rate": 0.0,
}

MODULE_KWARGS = {
    "example_input_array_shape": [1, 1, 1, 160, 160],
}

# GPU non-determinism tolerance for ConvNeXt convolutions.
# Tight enough to catch functional bugs, loose enough for hardware variance.
ATOL = 0.02
RTOL = 1e-2
# Minimum Pearson correlation to verify overall embedding agreement.
MIN_PEARSON_R = 0.999


def _build_module(checkpoint_path):
    """Build ContrastiveModule and load pretrained checkpoint."""
    encoder = ContrastiveEncoder(**ENCODER_KWARGS)
    module = ContrastiveModule(encoder=encoder, **MODULE_KWARGS)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    result = module.load_state_dict(ckpt["state_dict"])
    return module, result


@requires_hpc_and_gpu
@pytest.mark.hpc_integration
def test_checkpoint_loads_into_modular_contrastive_module(checkpoint_path):
    """INFER-01: Checkpoint loads without state dict key mismatches."""
    seed_everything(42)
    module, result = _build_module(checkpoint_path)

    assert len(result.missing_keys) == 0, f"Missing keys: {result.missing_keys}"
    assert len(result.unexpected_keys) == 0, f"Unexpected keys: {result.unexpected_keys}"

    x = torch.randn(1, 1, 1, 160, 160)
    module.eval()
    with torch.no_grad():
        features, projections = module(x)
    assert features.shape == (1, 768)
    assert projections.shape == (1, 32)


@requires_hpc_and_gpu
@pytest.mark.hpc_integration
def test_predict_embeddings_and_exact_match(
    tmp_path,
    checkpoint_path,
    reference_zarr_path,
    data_zarr_path,
    tracks_zarr_path,
):
    """INFER-02 + INFER-03: Predict writes embeddings and matches reference."""
    import anndata as ad

    from viscy_data.triplet import TripletDataModule
    from viscy_utils.callbacks.embedding_writer import EmbeddingWriter

    # --- Setup ---
    seed_everything(42)

    module, _ = _build_module(checkpoint_path)

    datamodule = TripletDataModule(
        data_path=str(data_zarr_path),
        tracks_path=str(tracks_zarr_path),
        source_channel=["Phase3D"],
        z_range=[0, 1],
        batch_size=64,
        num_workers=16,
        initial_yx_patch_size=[160, 160],
        final_yx_patch_size=[160, 160],
        normalizations=[
            NormalizeSampled(
                keys=["Phase3D"],
                level="fov_statistics",
                subtrahend="mean",
                divisor="std",
            )
        ],
    )

    output_path = tmp_path / "test_embeddings.zarr"
    writer = EmbeddingWriter(
        output_path=output_path,
        phate_kwargs=None,
        pca_kwargs=None,
        umap_kwargs=None,
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        precision="32-true",
        callbacks=[writer],
        inference_mode=True,
        enable_progress_bar=False,
        logger=False,
    )

    # --- INFER-02: Predict and write embeddings ---
    trainer.predict(module, datamodule=datamodule)

    assert output_path.exists(), f"Output zarr not written at {output_path}"

    pred = ad.read_zarr(output_path)
    assert pred.X.shape == (
        39170,
        768,
    ), f"Expected features (39170, 768), got {pred.X.shape}"
    assert "X_projections" in pred.obsm, "Missing X_projections in obsm"
    assert pred.obsm["X_projections"].shape == (
        39170,
        32,
    ), f"Expected projections (39170, 32), got {pred.obsm['X_projections'].shape}"

    # --- INFER-03: Numerical exactness against reference ---
    ref = ad.read_zarr(str(reference_zarr_path))

    # Correlation check: overall embedding agreement must be near-perfect
    r_features, _ = stats.pearsonr(pred.X.flatten(), ref.X.flatten())
    assert r_features > MIN_PEARSON_R, f"Feature Pearson r={r_features:.6f} < {MIN_PEARSON_R}"

    r_proj, _ = stats.pearsonr(
        pred.obsm["X_projections"].flatten(),
        ref.obsm["X_projections"].flatten(),
    )
    assert r_proj > MIN_PEARSON_R, f"Projection Pearson r={r_proj:.6f} < {MIN_PEARSON_R}"

    # Element-wise tolerance check
    np.testing.assert_allclose(
        pred.X,
        ref.X,
        rtol=RTOL,
        atol=ATOL,
        err_msg="Feature embeddings (X) exceed tolerance vs reference",
    )

    np.testing.assert_allclose(
        pred.obsm["X_projections"],
        ref.obsm["X_projections"],
        rtol=RTOL,
        atol=ATOL,
        err_msg="Projections (obsm/X_projections) exceed tolerance vs reference",
    )

    # Verify sample ordering is preserved
    pred_fov = pred.obs["fov_name"].values
    ref_fov = ref.obs["fov_name"].values
    np.testing.assert_array_equal(
        pred_fov,
        ref_fov,
        err_msg="FOV names do not match (sample ordering changed)",
    )

    pred_ids = pred.obs["id"].values
    ref_ids = ref.obs["id"].values
    np.testing.assert_array_equal(
        pred_ids,
        ref_ids,
        err_msg="Sample IDs do not match (sample ordering changed)",
    )
