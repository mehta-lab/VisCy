"""Segmentation workflows for evaluation."""

from pathlib import Path

import numpy as np
import torch

try:
    from segmenter_model_zoo.zoo import SegModel, SuperModel
except ImportError:
    SegModel = None  # type: ignore[assignment, misc]
    SuperModel = None  # type: ignore[assignment, misc]

try:
    from aicssegmentation.structure_wrapper.seg_lamp1 import Workflow_lamp1
    from aicssegmentation.structure_wrapper.seg_npm1 import Workflow_npm1
    from aicssegmentation.structure_wrapper.seg_npm1_SR import (
        Workflow_npm1_SR,  # noqa: F401
    )
    from aicssegmentation.structure_wrapper.seg_sec61b import Workflow_sec61b
    from aicssegmentation.structure_wrapper.seg_sec61b_dual import (
        Workflow_sec61b_dual,  # noqa: F401
    )
    from aicssegmentation.structure_wrapper.seg_tomm20 import Workflow_tomm20
except ImportError:
    Workflow_npm1 = None  # type: ignore[assignment, misc]
    Workflow_lamp1 = None  # type: ignore[assignment, misc]
    Workflow_sec61b = None  # type: ignore[assignment, misc]
    Workflow_tomm20 = None  # type: ignore[assignment, misc]

from cubic.cuda import ascupy, asnumpy
from cubic.skimage import filters as _cubic_filters

NUCLEUS_GAUSSIAN_SIGMA = 1.0
"""Isotropic Gaussian sigma (voxels) applied to the H2B input before
``structure_H2B_100x_hipsc.apply_on_single_zstack``.

The wrapper internally applies ``aicsmlsegment.simple_norm(1.5, 10)``,
which fits a Gaussian via MLE to the full intensity distribution and
stretches ``[m - 1.5σ, m + 10σ]`` to ``[0, 1]``. Bright chromatin tips and
shot-noise outliers in noisy GT fluorescence inflate the fitted σ, the
stretch range overshoots the actual signal range, and faint nuclei
collapse into the low end of ``[0, 1]`` where the model's confidence is
near zero. A σ=1 voxel Gaussian suppresses those outliers enough to bring
``simple_norm``'s fit back into the actual signal range. Tested on iPSC
GT and predictions where it roughly doubles the recovered nucleus mask
area; A549 is unaffected (already clean intensity statistics). Membrane
+ ER + mitochondria are not smoothed because their backends either
saturate (DL CAAX) or have their own internal preprocessing (classical
aicssegmentation workflows)."""


def _require_segmenter_model_zoo():
    if SuperModel is None:
        raise ImportError(
            "segmenter_model_zoo is required for nucleus/membrane segmentation. "
            "Install it with: pip install segmenter-model-zoo"
        )


def _require_aicssegmentation():
    if Workflow_npm1 is None:
        raise ImportError(
            "aicssegmentation is required for organelle segmentation workflows. "
            "Install it with: pip install aicssegmentation"
        )


def _smooth_nucleus_input(img, sigma: float = NUCLEUS_GAUSSIAN_SIGMA):
    """Apply isotropic Gaussian smoothing to dampen bright outliers in H2B input.

    Runs on GPU via ``cubic`` when CUDA is available, on CPU otherwise. The
    SuperModel inference downstream requires GPU regardless, so this is the
    same constraint surface. See ``NUCLEUS_GAUSSIAN_SIGMA`` docstring for the
    motivation.
    """
    img_dev = ascupy(img.astype(np.float32, copy=False))
    smoothed = _cubic_filters.gaussian(img_dev, sigma=sigma, preserve_range=True)
    return asnumpy(smoothed)


def segment(img, target_name=None, seg_model: "SuperModel" = None):
    """Run the organelle-specific segmentation workflow on a single z-stack.

    Parameters
    ----------
    img :
        3-D image array (Z, Y, X).
    target_name :
        Organelle name: one of ``nucleus``, ``membrane``, ``nucleoli``,
        ``lysosomes``, ``er``, ``mitochondria``.
    seg_model :
        Pre-loaded ``SuperModel`` required for nucleus/membrane segmentation.

    Returns
    -------
    numpy.ndarray
        Boolean mask with the same spatial shape as *img*.
    """
    if target_name in ["nucleus", "membrane"]:
        _require_segmenter_model_zoo()
        if seg_model is None:
            raise ValueError("seg_model (a loaded SuperModel) must be provided for nucleus and membrane segmentation.")
        if target_name == "nucleus":
            img = _smooth_nucleus_input(img)
        mask = seg_model.apply_on_single_zstack(img[None, ...])

    elif target_name == "nucleoli":
        _require_aicssegmentation()
        mask = Workflow_npm1(img, output_type="array")
    elif target_name == "lysosomes":
        _require_aicssegmentation()
        mask = Workflow_lamp1(img, output_type="array")
    elif target_name == "er":
        _require_aicssegmentation()
        mask = Workflow_sec61b(img, output_type="array")
    elif target_name == "mitochondria":
        _require_aicssegmentation()
        mask = Workflow_tomm20(img, output_type="array")
    else:
        raise ValueError(f"Unsupported target_name: {target_name}")

    return mask.astype(bool)


def prepare_segmentation_model(config):
    """Load and return the segmentation model specified in *config*.

    Returns ``None`` for organelles that use classical (non-DL) workflows.
    Respects ``config.use_gpu`` when deciding whether to move models to GPU.

    Returns ``None`` when ``io.require_complete_cache=true`` AND no per-FOV
    call path can invoke ``segment()`` without a cached mask. Concretely:
    organelle targets (``er``/``nucleoli``/``lysosomes``/``mitochondria``)
    delegate to aicssegmentation workflows that don't take ``seg_model``,
    so skipping the SuperModel load is safe. For ``nucleus``/``membrane``,
    the per-T loop in ``pipeline._process_one_fov`` falls back to
    ``segment(predict[t], seg_model=seg_model)`` whenever the pred-side
    mask cache is disabled (``io.pred_cache_dir=None``); in that case we
    must still load SuperModel even under ``require_complete_cache=true``
    or the fallback raises ``ValueError`` mid-loop.
    """
    require_complete = bool(getattr(config.io, "require_complete_cache", False))
    if config.target_name not in [
        "nucleus",
        "membrane",
        "nucleoli",
        "lysosomes",
        "er",
        "mitochondria",
    ]:
        raise ValueError(f"Invalid target_name in config: {config.target_name!r}")
    if require_complete:
        pred_cache_dir = getattr(config.io, "pred_cache_dir", None)
        if config.target_name not in ("nucleus", "membrane") or pred_cache_dir is not None:
            return None
    if config.target_name in ["nucleus", "membrane"]:
        _require_segmenter_model_zoo()
        if config.target_name == "nucleus":
            checkpoint_name = "structure_H2B_100x_hipsc"
        else:
            checkpoint_name = "structure_AAVS1_100x_hipsc"
        checkpoints_dir = Path(__file__).parent / "checkpoints"
        seg_model = SuperModel(checkpoint_name, {"local_path": str(checkpoints_dir)})
        use_gpu = getattr(config, "use_gpu", True)
        if use_gpu and torch.cuda.is_available():
            for m in seg_model.models:
                if isinstance(m, SegModel):
                    m.to_gpu("cuda")
    else:
        seg_model = None
    return seg_model
