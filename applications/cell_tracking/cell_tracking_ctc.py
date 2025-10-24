#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "tracksdata @ git+https://github.com/royerlab/tracksdata.git",
#   "onnxruntime-gpu",
#   "napari[pyqt5]",
#   "gurobipy",
#   "py-ctcmetrics",
#   "spatial-graph",
# ]
# ///

from pathlib import Path
import polars as pl
import numpy as np
import napari
import onnxruntime as ort

from dask.array.image import imread
from numpy.typing import NDArray
from toolz import curry
from rich import print
from scipy.ndimage import gaussian_filter

import tracksdata as td


def _seg_dir(dataset_dir: Path, dataset_num: str) -> Path:
    return dataset_dir / f"{dataset_num}_ERR_SEG"


def _pad(image: NDArray, shape: tuple[int, int], mode: str) -> NDArray:
    """
    Pad the image to the given shape.
    """
    diff = np.asarray(shape) - np.asarray(image.shape)

    if diff.sum() == 0:
        return image

    left = diff // 2
    right = diff - left

    return np.pad(image, tuple(zip(left, right)), mode=mode)


@curry
def _crop_embedding(
    frame: NDArray,
    mask: list[td.nodes.Mask],
    final_shape: tuple[int, int],
    session: ort.InferenceSession,
    input_name: str,
) -> NDArray[np.float32]:
    """
    Crop the frame and compute the DynaCLR embedding.

    Parameters
    ----------
    frame : NDArray
        The frame to crop.
    mask : Mask
        The mask to crop the frame.
    shape : tuple[int, int]
        The shape of the crop.
    session : ort.InferenceSession
        The session to use for the embedding.
    input_name : str
        The name of the input tensor.
    padding : int, optional
        The padding to apply to the crop.

    Returns
    -------
    NDArray[np.float32]
        The embedding of the crop.
    """
    label_img = np.zeros_like(frame, dtype=np.int16)
    
    crops = []
    for i, m in enumerate(mask, start=1):

        if frame.ndim == 3:
            crop_shape = (1, *final_shape)
        else:
            crop_shape = final_shape
        
        label_img[m.mask_indices()] = i

        crop = m.crop(frame, shape=crop_shape).astype(np.float32)
        crop_mask = (m.crop(label_img, shape=crop_shape) == i).astype(np.float32)

        if crop.ndim == 3:
            assert crop.shape[0] == 1, f"Expected 1 z-slice in 3D crop. Found {crop.shape[0]}"
            crop = crop[0]
            crop_mask = crop_mask[0]

        crop = _pad(crop, final_shape, mode="reflect")
        crop_mask = _pad(crop_mask, final_shape, mode="constant")

        blurred_mask = gaussian_filter(crop_mask, sigma=5)
        blurred_coef = blurred_mask.max()
        if blurred_coef > 1e-8:  # if too small use the binary mask
            crop_mask = np.maximum(crop_mask, blurred_mask / blurred_coef)

        mu, sigma = np.mean(crop), np.std(crop)
        # mu = np.median(crop)
        # sigma = np.quantile(crop, 0.99) - mu
        crop = (crop - mu) / np.maximum(sigma, 1e-8)

        # removing background
        crop = crop * crop_mask

        if crop.shape != final_shape:
            raise ValueError(f"Crop shape {crop.shape} does not match final shape {final_shape}")
        
        crops.append(crop)

    # expanding batch, channel, and z dimensions
    crops = np.stack(crops, axis=0)
    crops = crops[:, np.newaxis, np.newaxis, ...]
    output = session.run(None, {input_name: crops})

    # import napari
    # viewer = napari.Viewer()
    # viewer.add_image(frame)
    # viewer.add_image(np.squeeze(crops))
    # napari.run()

    # embedding = output[-1]   # projected 32-dimensional embedding
    embedding = output[0]  # 768-dimensional embedding
    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

    return [e for e in embedding]


def _add_dynaclr_attrs(
    model_path: Path,
    graph: td.graph.InMemoryGraph,
    images: NDArray,
) -> None:
    """
    Add DynaCLR embedding attributes to each node in the graph
    and compute the cosine similarity for existing edges.

    Parameters
    ----------
    graph : td.graph.InMemoryGraph
        The graph to add the attributes to.
    images : NDArray
        The images to use for the embedding.
    """

    session = ort.InferenceSession(model_path)

    input_name = session.get_inputs()[0].name
    input_dim = session.get_inputs()[0].shape
    input_type = session.get_inputs()[0].type

    print(f"Model input name: '{input_name}'")
    print(f"Expected input dimensions: {input_dim}")
    print(f"Expected input type: {input_type}")

    crop_attr_func = _crop_embedding(
        final_shape=(64, 64),
        session=session,
        input_name=input_name,
    )

    print("Adding DynaCLR embedding attributes ...")
    td.nodes.GenericFuncNodeAttrs(
        func=crop_attr_func,
        output_key="dynaclr_embedding",
        attr_keys=["mask"],
        batch_size=128,
    ).add_node_attrs(graph, frames=images)

    print("Adding cosine similarity attributes ...")
    td.edges.GenericFuncEdgeAttrs(
        func=np.dot,
        output_key="dynaclr_similarity",
        attr_keys="dynaclr_embedding",
    ).add_edge_attrs(graph)


def track_single_dataset(
    dataset_dir: Path,
    dataset_num: str,
    show_napari_viewer: bool,
    dynaclr_model_path: Path | None,
) -> None:
    """
    Main function to track cells in a dataset.

    Parameters
    ----------
    dataset_dir : Path
        Path to the dataset directory.
    dataset_num : str
        Number of the dataset.
    show_napari_viewer : bool
        Whether to show the napari viewer.
    dynaclr_model_path : Path | None
        Path to the DynaCLR model. If None, the model will not be used.
    """
    assert dataset_dir.exists(), f"Data directory {dataset_dir} does not exist."

    print(f"Loading labels from '{dataset_dir}'...")
    labels = imread(str(_seg_dir(dataset_dir, dataset_num) / "*.tif"))
    images = imread(str(dataset_dir / dataset_num / "*.tif"))

    gt_graph = td.graph.InMemoryGraph.from_ctc(dataset_dir / f"{dataset_num}_GT" / "TRA")

    print("Starting tracking ...")
    graph = td.graph.InMemoryGraph()

    nodes_operator = td.nodes.RegionPropsNodes()
    nodes_operator.add_nodes(graph, labels=labels)
    print(f"Number of nodes: {graph.num_nodes}")

    dist_operator = td.edges.DistanceEdges(
        distance_threshold=325.0, # 50,
        n_neighbors=10,
        delta_t=5, # 30,
    )
    dist_operator.add_edges(graph)
    print(f"Number of edges: {graph.num_edges}")

    td.edges.GenericFuncEdgeAttrs(
        func=lambda x, y: abs(x - y),
        output_key="delta_t",
        attr_keys="t",
    ).add_edge_attrs(graph)

    dist_weight = (-td.EdgeAttr(td.DEFAULT_ATTR_KEYS.EDGE_DIST) / dist_operator.distance_threshold).exp()

    if dynaclr_model_path is not None:
        _add_dynaclr_attrs(
            dynaclr_model_path,
            graph,
            images,
        )
        # decrease dynaclr similarity given the distance?
        edge_weight = -td.EdgeAttr("dynaclr_similarity") * dist_weight

    else:
        iou_operator = td.edges.IoUEdgeAttr(output_key="iou")
        iou_operator.add_edge_attrs(graph)

        edge_weight = -(td.EdgeAttr("iou") + 0.1) * dist_weight

    edge_weight = edge_weight / td.EdgeAttr("delta_t").clip(lower_bound=1)

    solver = td.solvers.ILPSolver(
        edge_weight=edge_weight,
        appearance_weight=0,
        disappearance_weight=0,
        division_weight=0.5,
        node_weight=-10,  # we assume all segmentations are correct
    )

    solution_graph = solver.solve(graph)

    print("Evaluating results ...")
    metrics = td.metrics.evaluate_ctc_metrics(
        solution_graph,
        gt_graph,
        input_reset=False,
        reference_reset=False,
    )

    if show_napari_viewer:
        print("Converting to napari format ...")
        tracks_df, track_graph, labels = td.functional.to_napari_format(
            graph, labels.shape, mask_key=td.DEFAULT_ATTR_KEYS.MASK
        )

        print("Opening napari viewer ...")
        viewer = napari.Viewer()
        viewer.add_image(images)
        viewer.add_labels(labels)
        viewer.add_tracks(tracks_df, graph=track_graph)
        napari.run()
    
    return metrics


def main() -> None:
    models = [
        Path("/hpc/projects/organelle_phenotyping/models/SEC61_TOMM20_G3BP1_Sensor/time_interval/dynaclr_gfp_rfp_ph_2D/deploy/dynaclr2d_classical_gfp_rfp_ph_temp0p5_batch128_ckpt146.onnx"),
        Path("/hpc/projects/organelle_phenotyping/models/SEC61_TOMM20_G3BP1_Sensor/time_interval/dynaclr_gfp_rfp_ph_2D/deploy/dynaclr2d_timeaware_gfp_rfp_ph_temp0p5_batch128_ckpt185.onnx"),
        Path("/hpc/projects/organelle_phenotyping/models/dynamorph_microglia/deploy/dynaclr2d_phase_brightfield_temp0p2_batch256_ckpt33.onnx"),
        Path("/hpc/projects/organelle_phenotyping/models/dynamorph_microglia/deploy/dynaclr2d_timeaware_phase_brightfield_temp0p2_batch256_ckpt13.onnx"),
        None,
    ]

    results = []

    dataset_root = Path("/hpc/reference/group.royer/CTC/training/")

    for model_path in models:
        for dataset_dir in sorted(dataset_root.iterdir()):

            for dataset_num in ["01", "02"]:
                # processing only datasets with segmentation (linking challenge)
                seg_dir = _seg_dir(dataset_dir, dataset_num)
                if not seg_dir.exists():
                    print(f"Skipping {dataset_dir.name} because it does not have segmentation")
                    continue
                
                metrics = track_single_dataset(
                    dataset_dir=dataset_dir,
                    dataset_num=dataset_num,
                    show_napari_viewer=False,
                    dynaclr_model_path=model_path,
                )
                metrics["model"] = "None" if model_path is None else model_path.stem
                metrics["dataset"] = dataset_dir.name
                metrics["dataset_num"] = dataset_num
                print(metrics)
                results.append(metrics)

                # update for every new result
                df = pl.DataFrame(results)
                df.write_csv("results.csv")

    print(
        df.group_by("model", "dataset").mean().select(
            "model", "dataset", "LNK", "BIO(0)", "OP_CLB(0)", "CHOTA"
        )
    )


if __name__ == "__main__":
    main()
