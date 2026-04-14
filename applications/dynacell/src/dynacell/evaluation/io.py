"""Image I/O utilities for evaluation."""

from contextlib import closing
from functools import partial
from pathlib import Path

import numpy as np
from iohub import read_images
from iohub.ngff import open_ome_zarr
from omegaconf import DictConfig

try:
    from skimage import io as skimage_io
except ImportError:
    skimage_io = None  # type: ignore[assignment]

try:
    from cubic.cuda import ascupy, asnumpy
    from cubic.skimage import transform
except ImportError:
    ascupy = None  # type: ignore[assignment]
    asnumpy = None  # type: ignore[assignment]
    transform = None  # type: ignore[assignment]


def _require_skimage():
    if skimage_io is None:
        raise ImportError("scikit-image is required for TIFF I/O. Install it with: pip install scikit-image")


def _require_cubic():
    if ascupy is None:
        raise ImportError("cubic is required for GPU array operations. Install it with: pip install cubic-s2")


def _is_zarr_path(path: Path) -> bool:
    """Return whether the input path points to a zarr store."""
    return path.suffix == ".zarr"


def _to_tczyx(image: np.ndarray) -> np.ndarray:
    """Convert image to TCZYX shape expected by OME-Zarr."""
    if image.ndim == 2:
        return image[np.newaxis, np.newaxis, np.newaxis, :, :]
    if image.ndim == 3:
        return image[np.newaxis, np.newaxis, :, :, :]
    if image.ndim == 4:
        return image[np.newaxis, :, :, :, :]
    if image.ndim == 5:
        return image
    raise ValueError(f"Unsupported image dimensions: {image.ndim}. Expected 2D to 5D image.")


def _read_ome_zarr(img_path: Path) -> np.ndarray:
    """Read image data from an OME-Zarr store."""
    with open_ome_zarr(img_path, mode="r") as dataset:
        if hasattr(dataset, "positions"):
            _, pos = next(dataset.positions())
            image = np.asarray(pos.data)
        else:
            image = np.asarray(dataset.data)
    return np.squeeze(image)


def _read_with_iohub(img_path: Path) -> np.ndarray:
    """Read image data from TIFF-like inputs via iohub readers."""
    with closing(read_images(img_path)) as reader:
        _, fov = next(iter(reader))
        image = np.asarray(fov[:])
    return np.squeeze(image)


def _save_ome_zarr(img_path: Path, image: np.ndarray) -> None:
    """Write image data to an OME-Zarr store."""
    image = _to_tczyx(image)
    channel_names = [f"channel_{idx}" for idx in range(image.shape[1])]
    with open_ome_zarr(img_path, layout="fov", mode="w", channel_names=channel_names) as dataset:
        dataset.create_image("0", image)


def _save_with_skimage(img_path: Path, image: np.ndarray) -> None:
    """Write image data to TIFF-like outputs via scikit-image."""
    _require_skimage()
    skimage_io.imsave(img_path, image, check_contrast=False)


def imread(img_path, use_gpu=False):
    """Read image from path."""
    _require_cubic()
    img_path = Path(img_path)
    if _is_zarr_path(img_path):
        image = _read_ome_zarr(img_path)
    else:
        image = _read_with_iohub(img_path)
    return ascupy(image) if use_gpu else asnumpy(image)


def imsave(img_path, image):
    """Save image to path."""
    _require_cubic()
    img_path = Path(img_path)
    image = asnumpy(image)
    if _is_zarr_path(img_path):
        _save_ome_zarr(img_path, image)
    else:
        _save_with_skimage(img_path, image)


def get_predict_transform(target_transform: str):
    """Return the appropriate transform function for predictions."""
    if target_transform in ("normalize", "norm_threshold"):
        return lambda x: x
    if target_transform == "norm_min_max":
        return partial(np.clip, a_min=0, a_max=1)
    raise ValueError(f"Unknown target transform {target_transform}")


def imread_predict(image_path, target_transform, use_gpu=True):
    """Load and transform a prediction image."""
    predict_transform = get_predict_transform(target_transform)
    image = imread(image_path, use_gpu=use_gpu)
    return predict_transform(image)


def preprocess_predictions(target, predict, preprocess_config: DictConfig):
    """Preprocess predictions according to configuration."""
    if "predict_threshold" in preprocess_config:
        threshold = preprocess_config.predict_threshold
        predict = np.where(predict > threshold, predict, 0)
    else:
        raise ValueError(f"Unknown preprocess config: {preprocess_config}")
    return target, predict


def load_target_bin(
    config: DictConfig,
    target_bin_path: Path,
    target_segment_gt: np.ndarray,
    target_shape: tuple,
    use_gpu: bool = False,
):
    """Load target binary mask based on configuration."""
    _require_cubic()
    if config.segment_gt_as_fg:
        target_bin = transform.resize(
            target_segment_gt,
            target_shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        )
    else:
        target_bin_stem = target_bin_path.stem.split("_bin")[0]
        target_bin_path = target_bin_path.with_name(f"{target_bin_stem}_bin{config.file_suffix}.tiff")
        target_bin = imread(target_bin_path, use_gpu=use_gpu)

    if config.binarize:
        target_bin = (target_bin > 0).astype(np.uint8)

    return target_bin


def load_predict_target(
    target_path: Path,
    target_bin_path: Path,
    predict_path: Path,
    target_transform: str,
    config: DictConfig,
):
    """Load and preprocess images for evaluation."""
    predict = imread_predict(predict_path, target_transform, config.use_gpu)
    target = imread(
        target_path.with_name(f"{target_path.stem}_{target_transform}.tiff"),
        config.use_gpu,
    )
    target_segment_gt = imread(target_path.with_name(f"{target_path.stem}_gt.tiff"), config.use_gpu)

    target_bin = load_target_bin(
        config.foreground,
        target_bin_path,
        target_segment_gt,
        target.shape,
        config.use_gpu,
    )

    if predict.shape != target.shape:
        raise ValueError(f"Prediction and image shapes do not match: {predict.shape} vs {target.shape}")
    if target.shape != target_bin.shape:
        raise ValueError(f"Image and binary mask shapes do not match: {target.shape} vs {target_bin.shape}")

    if "preprocess" in config and config.preprocess:
        target, predict = preprocess_predictions(target, predict, config.preprocess)

    return target, target_bin, target_segment_gt, predict
