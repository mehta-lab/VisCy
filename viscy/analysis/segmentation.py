import torch
import numpy as np
import click
from cellpose import models
from skimage.exposure import rescale_intensity, equalize_adapthist
from skimage.util import invert
from numpy.typing import ArrayLike


def nuc_mem_segmentation_cellposemodel_3D(
    czyx_data: ArrayLike, zyx_slicing: tuple[slice, slice, slice], **cellpose_kwargs
):
    """
    Segment nuclei and membranes using Cellpose 3D model.

    """

    Z_slice = zyx_slicing[0]
    Y_slice = zyx_slicing[1]
    X_slice = zyx_slicing[2]
    czyx_data = czyx_data[:, Z_slice, Y_slice, X_slice]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    segmentation_stack = np.zeros_like(czyx_data)
    click.echo(f"Segmentation Stack shape {segmentation_stack.shape}")
    cellpose_params = cellpose_kwargs["cellpose_kwargs"]
    c_idx = 0
    if "nucleus_kwargs" in cellpose_params:
        click.echo("Segmenting Nuclei")
        nuc_seg_kwargs = cellpose_params["nucleus_kwargs"]

        model_nucleus_3D = models.CellposeModel(
            model_type=cellpose_params["nuc_model_path"],
            # net_avg=True, #Note removed CP3.0
            gpu=True,
            device=torch.device(device),
        )
        nuc_segmentation, _, _ = model_nucleus_3D.eval(czyx_data, **nuc_seg_kwargs)
        segmentation_stack[c_idx] = nuc_segmentation.astype(np.uint16)
        c_idx += 1
    if "membrane_kwargs" in cellpose_params:
        click.echo("Segmenting Membrane")
        mem_seg_kwargs = cellpose_params["membrane_kwargs"]

        model_membrane_3D = models.CellposeModel(
            model_type=cellpose_params["mem_model_path"],
            # net_avg=True,
            gpu=True,
            device=torch.device(device),
        )
        c_idx_mem, c_idx_nuc = mem_seg_kwargs["channels"]
        mem_segmentation, _, _ = model_membrane_3D.eval(czyx_data, **mem_seg_kwargs)
        segmentation_stack[c_idx] = mem_segmentation.astype(np.uint16)

    return segmentation_stack


def nuc_mem_cp_segmentation_clahe_3D(
    czyx_data: ArrayLike, zyx_slicing: tuple, clahe_kwargs, **cellpose_kwargs
):
"""
    Segment nuclei and membranes using Cellpose 3D model with CLAHE applied to the input data.
"""

    Z_slice = zyx_slicing[0]
    Y_slice = zyx_slicing[1]
    X_slice = zyx_slicing[2]
    czyx_data = czyx_data[:, Z_slice, Y_slice, X_slice]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    segmentation_stack = np.zeros_like(czyx_data, dtype=np.uint16)
    click.echo(f"Segmentation Stack shape {segmentation_stack.shape}")
    cellpose_params = cellpose_kwargs["cellpose_kwargs"]
    # clahe_kwargs = clahe_kwargs['clahe']
    c_idx = 0
    if "nucleus_kwargs" in cellpose_params:
        click.echo("Segmenting Nuclei")
        nuc_seg_kwargs = cellpose_params["nucleus_kwargs"]

        model_nucleus_3D = models.CellposeModel(
            model_type=cellpose_params["nuc_model_path"],
            # net_avg=True, #Note removed CP3.0
            gpu=True,
            device=torch.device(device),
        )
        # Apply CLAHE before cellpose
        if "clahe_nuc" in clahe_kwargs:
            click.echo("Applying CLAHE to Nuclei")
            nuc_clahe = clahe_kwargs["clahe_nuc"]
            czyx_data[c_idx] = rescale_intensity(czyx_data[c_idx], out_range=(0.0, 1.0))
            czyx_data[c_idx] = equalize_adapthist(czyx_data[c_idx], **nuc_clahe)
        nuc_segmentation, _, _ = model_nucleus_3D.eval(czyx_data, **nuc_seg_kwargs)
        segmentation_stack[c_idx] = nuc_segmentation.astype(np.uint16)
        c_idx += 1
    if "membrane_kwargs" in cellpose_params:
        click.echo("Segmenting Membrane")
        mem_seg_kwargs = cellpose_params["membrane_kwargs"]

        if "clahe_mem" in clahe_kwargs:
            click.echo("Applying CLAHE to Membrane")
            mem_clahe = clahe_kwargs["clahe_mem"]
            czyx_data[c_idx] = rescale_intensity(
                invert(czyx_data[c_idx]), out_range=(0.0, 1.0)
            )
            czyx_data[c_idx] = equalize_adapthist(czyx_data[c_idx], **mem_clahe)
        model_membrane_3D = models.CellposeModel(
            model_type=cellpose_params["mem_model_path"],
            # net_avg=True,
            gpu=True,
            device=torch.device(device),
        )
        c_idx_mem, c_idx_nuc = mem_seg_kwargs["channels"]
        mem_segmentation, _, _ = model_membrane_3D.eval(czyx_data, **mem_seg_kwargs)
        segmentation_stack[c_idx] = mem_segmentation.astype(np.uint16)

    return segmentation_stack
