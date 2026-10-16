"""CellposeDense (differentiable Cellpose-SAM) against stock cellpose, and as a SegConsistencyLoss segmenter.

The network is ViT-L: a 256-px tile takes ~27 s forward on 6 CPU cores (measured), so the
tests that run it need a GPU and the stock ``cpsam`` checkpoint in cellpose's model dir.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("cellpose")

from cellpose import models as cp_models  # noqa: E402
from cellpose import transforms as cp_transforms  # noqa: E402
from generate_spotlight_v2_leaves import SEG_CONSISTENCY_ARGS  # noqa: E402

from dynacell.cellpose_dense import CellposeDense, normalize99_diff  # noqa: E402
from dynacell.engine import DynacellUNet  # noqa: E402
from viscy_utils.losses import SegConsistencyLoss  # noqa: E402

needs_cpsam = pytest.mark.skipif(
    not torch.cuda.is_available() or not (cp_models.MODEL_DIR / "cpsam").is_file(),
    reason="needs a GPU and the cpsam checkpoint in cellpose's model directory",
)


def _nuclei_image(shape: tuple[int, int], seed: int = 0) -> np.ndarray:
    """Bright disks of varying intensity on noise, in arbitrary units."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    img = np.zeros(shape, np.float32)
    for _ in range(14):
        cy, cx, r = rng.uniform(15, shape[0] - 15), rng.uniform(15, shape[1] - 15), rng.uniform(8, 16)
        img += ((yy - cy) ** 2 + (xx - cx) ** 2 < r**2) * rng.uniform(0.5, 1.0)
    return (300.0 * img + 100.0 + rng.normal(0, 10.0, shape)).astype(np.float32)


@pytest.fixture(scope="module")
def cpsam_bf16() -> CellposeDense:
    return CellposeDense("cpsam").cuda()


def test_configured_crop_is_one_cellpose_tile():
    """The segcons leaves' crop pads to exactly one tile; a 256-px crop would pad to 2 x 2."""
    crop = SEG_CONSISTENCY_ARGS["crop_size"]
    _, coords = CellposeDense.tiling(crop, crop)
    assert len(coords) == 1
    assert len(CellposeDense.tiling(256, 256)[1]) == 4


@needs_cpsam
def test_train_keeps_the_network_in_eval(cpsam_bf16):
    """Even as a registered submodule, a parent's train() leaves cellpose's network in eval mode."""
    parent = torch.nn.ModuleDict({"seg": cpsam_bf16})
    parent.train()
    assert not cpsam_bf16.training and not any(m.training for m in cpsam_bf16.net.modules())


@pytest.mark.skipif(not (cp_models.MODEL_DIR / "cpsam").is_file(), reason="needs the cpsam checkpoint")
def test_construction_leaves_the_global_torch_rng_untouched():
    """Building the segmenter must not shift the seeded torch RNG the datamodule's split draws from.

    ``HCSDataModule`` splits train/val FOVs with ``torch.randperm``; otherwise a segcons arm and
    its same-seed control would train on different splits.
    """
    torch.manual_seed(42)
    reference = torch.randperm(50)
    torch.manual_seed(42)
    CellposeDense("cpsam")
    assert torch.equal(reference, torch.randperm(50))


def test_refuses_the_evaluation_segmenter_and_missing_models():
    with pytest.raises(ValueError, match="evaluation segmenter"):
        CellposeDense("cpdino")
    with pytest.raises(FileNotFoundError, match="not found"):
        CellposeDense("no_such_cellpose_model")
    with pytest.raises(ValueError, match="checkpoint_tiles"):
        CellposeDense("cpsam", checkpoint_tiles=0)


def test_normalization_matches_cellpose_normalize99():
    img = _nuclei_image((96, 80))
    ours = normalize99_diff(torch.from_numpy(img)[None, None])[0, 0].numpy()
    np.testing.assert_allclose(ours, cp_transforms.normalize99(img), atol=1e-5)
    flat = normalize99_diff(torch.full((1, 1, 8, 8), 3.0))
    assert torch.equal(flat, torch.zeros_like(flat))


@needs_cpsam
def test_dense_outputs_match_stock_cellpose(cpsam_bf16):
    """Same flows and cell-probability logits as ``CellposeModel.eval`` on the same normalized input.

    Two overlapping tiles exercise the taper blend. The input is normalized once and fed to
    both: in bfloat16 the ~1e-6 float differences between the two normalize99 implementations
    move single outputs by up to 0.2 (measured), which says nothing about the tiling path.
    """
    img = cp_transforms.normalize99(_nuclei_image((256, 300))).astype(np.float32)
    with torch.no_grad():
        out = cpsam_bf16.dense_normalized(torch.from_numpy(img)[None, None].cuda())
    stock = cp_models.CellposeModel(gpu=True, pretrained_model=str(cp_models.MODEL_DIR / "cpsam"))
    _, flows, _ = stock.eval(img, normalize=False, diameter=None, channel_axis=None, compute_masks=False)
    np.testing.assert_allclose(out["flow"][0].cpu().numpy(), flows[1], atol=1e-5)
    np.testing.assert_allclose(out["cellprob_logit"][0, 0].cpu().numpy(), flows[2], atol=1e-5)
    assert np.abs(flows[1]).mean() > 0.05  # the image has cells, so the parity is not 0 == 0


@needs_cpsam
def test_input_gradient_matches_finite_difference():
    """Central difference on the post-normalization path (statistics are stop-gradient), TF32 off."""
    seg = CellposeDense("cpsam", use_bfloat16=False).cuda()
    tf32 = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        g = torch.Generator(device="cuda").manual_seed(0)
        x = torch.rand(1, 1, 64, 64, device="cuda", generator=g).requires_grad_(True)
        out = seg.dense_normalized(x)
        w = {k: torch.randn(out[k].shape, device="cuda", generator=g) for k in out}

        def f(z: torch.Tensor) -> torch.Tensor:
            o = seg.dense_normalized(z)
            return sum((o[k].double() * w[k]).sum() for k in o)

        f(x).backward()
        v = torch.randn(x.shape, device="cuda", generator=g)
        v = v / v.norm() * x.detach().norm() * 1e-2
        with torch.no_grad():
            num = float((f(x.detach() + v) - f(x.detach() - v)) / 2)
        ana = float((x.grad * v).sum())
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = tf32
    assert torch.isfinite(x.grad).all()
    assert abs(num - ana) / abs(num) < 1e-2


@needs_cpsam
def test_activation_checkpointing_keeps_outputs_and_gradient():
    """Checkpointed single tiles equal a stored pass over the same single tiles (2 tiles).

    Same grouping on both sides: regrouping tiles alone moves the input gradient by 5e-4
    (TF32) / 1.2e-5 (fp32) relative norm (measured), from the batched kernels, not checkpointing.
    """
    img = torch.from_numpy(_nuclei_image((64, 300)))[None, None].cuda()
    results = []
    for tiles in (1, None):
        seg = CellposeDense("cpsam", use_bfloat16=False, checkpoint_tiles=tiles).cuda()
        x = img.clone().requires_grad_(True)
        out = seg(x, batch_size=1)
        (out["flow"].square().sum() + out["cellprob_logit"].sum()).backward()
        results.append((out["flow"].detach(), x.grad))
        del seg
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    torch.testing.assert_close(results[0][1], results[1][1], rtol=0, atol=0)
    assert results[0][1].abs().sum() > 0


@needs_cpsam
def test_seg_consistency_through_cpsam_trains_a_backbone(cpsam_bf16):
    """Zero at pred == target; one DynacellUNet step under bf16 autocast puts its gradient on the backbone."""
    target = torch.from_numpy(np.stack([_nuclei_image((64, 64), seed=s) for s in range(4)])).reshape(2, 1, 2, 64, 64)
    target = target.repeat_interleave(2, dim=2).cuda()  # (2, 1, 4, 64, 64)
    loss_fn = SegConsistencyLoss(cpsam_bf16, crop_size=64)
    # The loss freezes its segmenter, and the LightningModule's train() cannot reach it.
    assert not cpsam_bf16.training and not any(p.requires_grad for p in cpsam_bf16.parameters())
    assert float(loss_fn(target.clone(), target)) == pytest.approx(0.0, abs=1e-6)

    torch.manual_seed(0)
    module = DynacellUNet(
        architecture="FNet3D",
        model_config={"in_channels": 1, "out_channels": 1, "depth": 1, "mult_chan": 8, "in_stack_depth": 4},
        seg_consistency=loss_fn,
        seg_consistency_weight=1.0,
        log_batches_per_epoch=0,
    ).cuda()
    module.train()
    assert not cpsam_bf16.training
    module.log = lambda *args, **kwargs: None
    batch = {"source": torch.randn_like(target), "target": target}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = module(batch["source"])
        segcons = loss_fn(pred, batch["target"])
    assert float(segcons.detach()) > 0
    segcons.backward()
    grads = [p.grad for p in module.model.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0
    assert all(p.grad is None for p in cpsam_bf16.parameters())
