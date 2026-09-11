"""Cellpose segmentation for OME-Zarr stores.

Note: This module imports torch, cellpose, and iohub lazily inside the
function body. This is a documented exception to the "import at top of
file" convention for optional heavyweight GPU dependencies.
"""

from __future__ import annotations

from pathlib import Path


def run_cellpose_segmentation(
    data_dir: Path,
    save_dir: Path,
    input_channel_names: list[str],
    *,
    do_3d: bool = True,
    use_gpu: bool = True,
    output_channel_names: list[str] | None = None,
    output_dtype: str = "uint8",
    zarr_layout: str = "hcs",
    zarr_version: str = "0.5",
    batch_size: int = 32,
    z_axis: int = 1,
    flow_3d_smooth: int = 10,
    niter_2d: int = 1000,
) -> None:
    """Run Cellpose segmentation on all positions in an OME-Zarr store.

    When ``do_3d`` is True, runs 3D evaluation with ``z_axis``,
    ``batch_size``, and ``flow_3d_smooth``.  When False, performs
    max-projection to 2D, evaluates with ``niter_2d`` iterations,
    and repeats masks across depth.

    Parameters
    ----------
    data_dir : Path
        Path to the input OME-Zarr store.
    save_dir : Path
        Path for the output segmentation zarr.
    input_channel_names : list[str]
        Channel names to select for segmentation input.
    do_3d : bool
        If True, run 3D Cellpose. If False, run 2D with max-projection.
    use_gpu : bool
        Whether to use GPU (if available).
    output_channel_names : list[str] | None
        Channel names for output zarr. Defaults to ``["segmentation"]``.
    output_dtype : str
        NumPy dtype string for output masks.
    zarr_layout : str
        OME-Zarr layout (default ``"hcs"``).
    zarr_version : str
        Zarr format version.
    batch_size : int
        Batch size for 3D evaluation.
    z_axis : int
        Z axis index for 3D evaluation.
    flow_3d_smooth : int
        Flow smoothing for 3D evaluation.
    niter_2d : int
        Number of iterations for 2D evaluation.
    """
    import numpy as np
    import torch
    from cellpose import models
    from iohub.ngff import open_ome_zarr
    from tqdm import tqdm

    _use_gpu = use_gpu and torch.cuda.is_available()
    model = models.CellposeModel(gpu=_use_gpu)

    if output_channel_names is None:
        output_channel_names = ["segmentation"]
    _output_dtype = np.dtype(output_dtype)

    with open_ome_zarr(
        save_dir,
        mode="w",
        layout=zarr_layout,
        channel_names=output_channel_names,
        version=zarr_version,
    ) as segmentation_results:
        with open_ome_zarr(data_dir, mode="r") as plate:
            positions = list(plate.positions())
            for pos_name, pos in tqdm(
                positions,
                desc=f"Processing {data_dir}",
            ):
                input_channel_indexes = [pos.get_channel_index(name) for name in input_channel_names]
                movie = pos.data[:, input_channel_indexes]
                T, _, D, H, W = movie.shape

                imgs = [movie[t] for t in range(T)]

                if do_3d:
                    masks, _, _ = model.eval(
                        imgs,
                        z_axis=z_axis,
                        channel_axis=0,
                        batch_size=batch_size,
                        do_3D=True,
                        flow3D_smooth=flow_3d_smooth,
                    )
                else:
                    imgs_2d = [np.max(img, axis=1) for img in imgs]
                    masks, _, _ = model.eval(imgs_2d, channel_axis=0, niter=niter_2d)
                    masks = [np.repeat(mask[None, ...], D, axis=0) for mask in masks]

                stacked = np.stack(masks, axis=0)[:, None, ...]
                # Cellpose returns an int32 LABEL image, so a narrowing cast
                # wraps modulo the dtype range instead of clipping: at
                # output_dtype="uint8" (the default) a FOV with >=256 objects
                # silently turns label 256 into background and merges 257 with
                # a distant cell. The shape guard below cannot see that, and
                # downstream instance-AP would score the corrupted field.
                max_label = int(stacked.max())
                dtype_max = np.iinfo(_output_dtype).max
                if max_label > dtype_max:
                    raise ValueError(
                        f"{pos_name}: Cellpose produced {max_label} labels, which does not fit "
                        f"output_dtype={_output_dtype!r} (max {dtype_max}); labels would wrap "
                        "silently. Pass output_dtype='uint16'."
                    )
                masks_arr = stacked.astype(_output_dtype)

                t, _, d, h, w = masks_arr.shape
                if (t, d, h, w) != (T, D, H, W):
                    raise ValueError(
                        f"Segmentation output shape {(t, d, h, w)} does not match input shape {(T, D, H, W)}"
                    )

                row, col, fov = pos_name.split("/")
                seg_pos = segmentation_results.create_position(row, col, fov)
                seg_pos.create_image("0", masks_arr)
