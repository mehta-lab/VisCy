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

from rich import print

import tracksdata as td
from cell_tracking_ctc import _track


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

    model_path = Path("/hpc/projects/organelle_phenotyping/models/SEC61_TOMM20_G3BP1_Sensor/deploy/dynaclr2d_timeaware_bagchannels_160patch_ckpt104.onnx")

    graph, solution_graph = _track(
        model_path,
        img,
        segm,
        dist_edge_kwargs={"delta_t": 1},
        ilp_kwargs={"division_weight": 0.95},
    )

    gt_tracks_df, gt_track_graph, gt_labels = td.functional.to_napari_format(
        gt_graph,
        segm.shape,
        solution_key=None,
        mask_key=td.DEFAULT_ATTR_KEYS.MASK,
    )

    tracks_df, track_graph, labels = td.functional.to_napari_format(
        solution_graph,
        segm.shape,
        solution_key=None,
        mask_key=td.DEFAULT_ATTR_KEYS.MASK,
    )

    metrics = td.metrics.evaluate_ctc_metrics(
        gt_graph,
        solution_graph,
        input_reset=False,
        reference_reset=False,
    )

    print(metrics)

    viewer = napari.Viewer()
    viewer.add_image(img)
    viewer.add_labels(segm, name="GT labels")
    viewer.add_tracks(gt_tracks_df, graph=gt_track_graph, name="GT tracks")
    viewer.add_tracks(tracks_df, graph=track_graph, name="Solution tracks")
    viewer.add_labels(labels, name="Solution labels")

    # solution_graph.match(gt_graph)
    # td.metrics.visualize_matches(
    #     solution_graph,
    #     gt_graph,
    #     viewer=viewer,
    # )

    napari.run()


if __name__ == "__main__":
    main()
