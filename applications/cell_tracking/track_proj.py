#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "tracksdata @ git+https://github.com/royerlab/tracksdata.git",
#   "contrastive-td @ git+https://github.com/royerlab/contrastive-td.git",
#   "onnxruntime-gpu",
#   "napari[pyqt5]",
#   "gurobipy",
#   "py-ctcmetrics==1.2.2",
#   "spatial-graph",
# ]
# ///

from pathlib import Path
import polars as pl
import napari
import zarr
import dask.array as da
from tracksdata.io._ctc import _add_edges_from_tracklet_ids

from dask.array.image import imread
from rich import print

import tracksdata as td
from cell_tracking_ctc import _add_dynaclr_attrs


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


def _load_gt_graph(
    segm: da.Array,
    tracklets_df: pl.DataFrame,
) -> td.graph.InMemoryGraph:

    tracklets_df = tracklets_df.filter(pl.col("parent_track_id") != -1)
    tracklet_id_graph = dict(zip(
        tracklets_df["track_id"].to_list(),
        tracklets_df["parent_track_id"].to_list(),
    ))

    gt_graph = td.graph.InMemoryGraph()

    td.nodes.RegionPropsNodes(
        extra_properties=["label"],
    ).add_nodes(gt_graph, labels=segm)

    _add_edges_from_tracklet_ids(
        gt_graph,
        gt_graph.node_attrs(attr_keys=[
            td.DEFAULT_ATTR_KEYS.NODE_ID,
            td.DEFAULT_ATTR_KEYS.T,
            "label",
        ]),
        tracklet_id_graph=tracklet_id_graph,
        tracklet_id_key="label",
    )

    return gt_graph


def main() -> None:

    img_dir = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_08_26_A549_SEC61_TOMM20_ZIKV/4-phenotyping/0-train-test/2025_08_26_A549_SEC61_TOMM20_ZIKV.zarr")

    segm_dir = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_08_26_A549_SEC61_TOMM20_ZIKV/2-assemble/tracking_annotation.zarr")

    pos_key = "A/2/000000/0"
    img_ds = zarr.open(img_dir, mode="r")
    img = da.from_zarr(img_ds[pos_key])[:, 0, 0]

    segm_ds = zarr.open(segm_dir, mode="r")
    segm = da.from_zarr(segm_ds[pos_key])[:, 0, 0]

    print(img.shape)
    print(segm.shape)

    short_pos_key = pos_key[:-2]
    suffix = "_".join(short_pos_key.split("/"))

    tracklets_graph_path = segm_dir / short_pos_key / f"tracks_{suffix}.csv"
    tracklets_df = pl.read_csv(tracklets_graph_path)
    gt_graph = _load_gt_graph(segm, tracklets_df)

    tracks_df, track_graph, labels = td.functional.to_napari_format(
        gt_graph,
        segm.shape,
        solution_key=None,
        mask_key=td.DEFAULT_ATTR_KEYS.MASK,
    )

    viewer = napari.Viewer()
    viewer.add_image(img)
    viewer.add_labels(segm)
    viewer.add_tracks(tracks_df, graph=track_graph)
    napari.run()



    return
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
