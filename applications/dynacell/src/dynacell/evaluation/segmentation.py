"""Segmentation workflows for evaluation."""

from pathlib import Path

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
            raise ValueError("SegModel must be provided for nucleus and membrane segmentation.")
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
    """
    if config.target_name not in [
        "nucleus",
        "membrane",
        "nucleoli",
        "lysosomes",
        "er",
        "mitochondria",
    ]:
        raise ValueError(f"Invalid target_name in config: {config.target_name!r}")
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
