"""CLI tool for CTC tracking accuracy benchmarking with DynaCLR embeddings.

Evaluates how well DynaCLR embedding similarity, used as an additional edge cost,
improves cell tracking accuracy on CTC (Cell Tracking Challenge) benchmark datasets.

For each (ONNX model, CTC dataset, sequence) combination:
1. Load segmentation masks and raw images.
2. Build a tracksdata graph (nodes from masks, candidate edges via DistanceEdges).
3. If a model is provided, run ONNX inference on cell crops and weight edges by
   embedding cosine similarity * spatial distance weight.
4. If no model is provided, use IoU + spatial distance (baseline).
5. Solve the tracking with ILP and evaluate against CTC ground truth.

Usage
-----
dynaclr evaluate-tracking-accuracy -c tracking_accuracy_config.yaml
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import click
import numpy as np
import polars as pl
import tracksdata as td
from dask.array.image import imread
from numpy.typing import NDArray
from rich import print as rprint
from skimage.transform import resize

from dynaclr.evaluation.benchmarking.tracking_accuracy.config import (
    CTCDatasetEntry,
    ONNXModelEntry,
    TrackingAccuracyConfig,
)
from dynaclr.evaluation.benchmarking.tracking_accuracy.utils import (
    normalize_crop,
    pad_to_shape,
    seg_dir,
)
from viscy_utils.cli_utils import load_config

_logger = logging.getLogger(__name__)


def _load_ctc_metadata(path: Path) -> dict[str, float]:
    """Load dataset name → x pixel size (µm) from Jordao's CTC metadata YAML.

    Format: ``dataset_name: [interval_min, y_um, x_um]``

    Parameters
    ----------
    path : Path
        Path to the metadata YAML file.

    Returns
    -------
    dict[str, float]
        Mapping from dataset name to x pixel size in µm.
    """
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f)
    # value is [interval_min, y_um, x_um] — take x (index 2)
    return {name: values[2] for name, values in raw.items() if isinstance(values, list)}


def _crop_embedding(
    frame: NDArray,
    mask: list,
    source_shape: tuple[int, int],
    final_shape: tuple[int, int],
    session: Any,
    input_name: str,
) -> list[NDArray]:
    """Crop cells from a frame and compute DynaCLR embeddings via ONNX.

    Parameters
    ----------
    frame : NDArray
        Raw image frame (2-D or 3-D with a single z-slice).
    mask : list[td.nodes.Mask]
        Cell masks for this frame. The parameter name must match the graph
        attribute key (``"mask"`` in ``attr_keys``).
    source_shape : tuple[int, int]
        (height, width) to extract from the image in dataset pixels.
        If different from ``final_shape``, the crop is resized to ``final_shape``
        to correct for pixel size differences between dataset and training data.
    final_shape : tuple[int, int]
        (height, width) of the model input (must match ONNX input size).
    session : ort.InferenceSession
        ONNX runtime inference session.
    input_name : str
        Name of the ONNX model's input tensor.

    Returns
    -------
    list[NDArray]
        L2-normalized embedding vector for each mask (same order).
    """
    # Compute frame-level stats once — matches timepoint_statistics normalization used in training
    frame_f32 = frame.astype(np.float32)
    frame_mean = float(np.mean(frame_f32))
    frame_std = float(np.std(frame_f32))

    label_img = np.zeros_like(frame, dtype=np.int16)
    crops = []

    for i, m in enumerate(mask, start=1):
        if frame.ndim == 3:
            extract_shape = (1, *source_shape)
        else:
            extract_shape = source_shape

        label_img[m.mask_indices()] = i

        crop = m.crop(frame, shape=extract_shape).astype(np.float32)

        if crop.ndim == 3:
            if crop.shape[0] != 1:
                raise ValueError(f"Expected 1 z-slice in 3D crop, got {crop.shape[0]}")
            crop = crop[0]

        crop = pad_to_shape(crop, source_shape, mode="reflect")

        if source_shape != final_shape:
            crop = resize(crop, final_shape, order=1, anti_aliasing=True, preserve_range=True).astype(np.float32)

        crop = normalize_crop(crop, frame_mean, frame_std)

        if crop.shape != final_shape:
            raise ValueError(f"Crop shape {crop.shape} != final_shape {final_shape}")

        crops.append(crop)

    # shape: (batch, channel, z, h, w)
    batch = np.stack(crops, axis=0)[:, np.newaxis, np.newaxis, ...]
    output = session.run(None, {input_name: batch})

    embeddings = output[0]  # backbone features (e.g. 768-dim)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return list(embeddings)


def _add_dynaclr_attrs(
    model_path: Path,
    graph: td.graph.InMemoryGraph,
    images: NDArray,
    model_input_shape: tuple[int, int],
    batch_size: int,
    pixel_size_scale: float,
) -> None:
    """Add DynaCLR embedding node attributes and cosine similarity edge attributes.

    Parameters
    ----------
    model_path : Path
        Path to the exported ONNX model.
    graph : td.graph.InMemoryGraph
        Graph with nodes already added (must have ``mask`` attribute).
    images : NDArray
        Raw image stack, shape (T, H, W) or (T, Z, H, W).
    model_input_shape : tuple[int, int]
        (height, width) of the ONNX model input (e.g. (160, 160)).
    batch_size : int
        Number of crops per ONNX inference call.
    pixel_size_scale : float
        Ratio of dataset pixel size to model training pixel size
        (dataset_um / model_um). Crops are extracted at
        ``model_input_shape * pixel_size_scale`` and resized to ``model_input_shape``.
        Use 1.0 when no rescaling is needed.
    """
    import onnxruntime as ort

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    _logger.info(
        "ONNX model: input='%s' shape=%s type=%s",
        input_name,
        session.get_inputs()[0].shape,
        session.get_inputs()[0].type,
    )

    source_shape = (
        round(model_input_shape[0] * pixel_size_scale),
        round(model_input_shape[1] * pixel_size_scale),
    )
    _logger.info(
        "Crop pipeline: extract %s px -> resize to %s px (scale=%.3f)",
        source_shape,
        model_input_shape,
        pixel_size_scale,
    )

    from toolz import curry

    crop_fn = curry(_crop_embedding)(
        source_shape=source_shape,
        final_shape=model_input_shape,
        session=session,
        input_name=input_name,
    )

    graph.add_node_attr_key("dynaclr_embedding", dtype=pl.List(pl.Float32))

    td.nodes.GenericFuncNodeAttrs(
        func=crop_fn,
        output_key="dynaclr_embedding",
        attr_keys=["mask"],
        batch_size=batch_size,
    ).add_node_attrs(graph, frames=images)

    td.edges.GenericFuncEdgeAttrs(
        func=np.dot,
        output_key="dynaclr_similarity",
        attr_keys="dynaclr_embedding",
    ).add_edge_attrs(graph)


def _build_and_solve(
    model_path: Path | None,
    images: NDArray,
    labels: NDArray,
    config: TrackingAccuracyConfig,
    pixel_size_scale: float = 1.0,
) -> tuple[td.graph.InMemoryGraph, td.graph.InMemoryGraph]:
    """Build a tracksdata graph and solve tracking.

    Parameters
    ----------
    model_path : Path or None
        ONNX model path. None uses the IoU + spatial baseline.
    images : NDArray
        Raw image stack (T, H, W).
    labels : NDArray
        Segmentation label stack (T, H, W).
    config : TrackingAccuracyConfig
        Evaluation configuration.
    pixel_size_scale : float
        Ratio of dataset pixel size to model training pixel size
        (dataset_um / model_um). Passed to ``_add_dynaclr_attrs``. Default 1.0.

    Returns
    -------
    graph : td.graph.InMemoryGraph
        Full candidate graph (all nodes + candidate edges).
    solution_graph : td.graph.InMemoryGraph
        ILP-solved tracking result.
    """
    graph = td.graph.InMemoryGraph()

    td.nodes.RegionPropsNodes().add_nodes(graph, labels=labels)
    _logger.info("Nodes: %d", graph.num_nodes())

    dist_op = td.edges.DistanceEdges(
        distance_threshold=config.distance_threshold,
        n_neighbors=config.n_neighbors,
        delta_t=config.delta_t,
    )
    dist_op.add_edges(graph)
    _logger.info("Candidate edges: %d", graph.num_edges())

    td.edges.GenericFuncEdgeAttrs(
        func=lambda x, y: abs(x - y),
        output_key="delta_t",
        attr_keys="t",
    ).add_edge_attrs(graph)

    dist_weight = (-td.EdgeAttr(td.DEFAULT_ATTR_KEYS.EDGE_DIST) / config.distance_threshold).exp()

    if model_path is not None:
        _add_dynaclr_attrs(model_path, graph, images, config.model_input_shape, config.batch_size, pixel_size_scale)
        edge_weight = -td.EdgeAttr("dynaclr_similarity") * dist_weight
    else:
        td.edges.IoUEdgeAttr(output_key="iou").add_edge_attrs(graph)
        edge_weight = -(td.EdgeAttr("iou") + 0.1) * dist_weight

    edge_weight = edge_weight / td.EdgeAttr("delta_t").clip(lower_bound=1)

    solver = td.solvers.ILPSolver(
        edge_weight=edge_weight,
        appearance_weight=config.appearance_weight,
        disappearance_weight=config.disappearance_weight,
        division_weight=config.division_weight,
        node_weight=config.node_weight,
    )
    solution_graph = solver.solve(graph)

    return graph, solution_graph


def _show_napari_viewer(
    graph: td.graph.InMemoryGraph,
    images: NDArray,
    labels: NDArray,
) -> None:
    """Open a napari viewer with the tracking result overlaid on the raw images.

    Parameters
    ----------
    graph : td.graph.InMemoryGraph
        Full candidate graph (used to derive napari tracks format).
    images : NDArray
        Raw image stack (T, H, W).
    labels : NDArray
        Segmentation label stack (T, H, W).
    """
    import napari

    tracks_df, track_graph, label_stack = td.functional.to_napari_format(
        graph, labels.shape, mask_key=td.DEFAULT_ATTR_KEYS.MASK
    )
    viewer = napari.Viewer()
    viewer.add_image(images)
    viewer.add_labels(label_stack)
    viewer.add_tracks(tracks_df, graph=track_graph)
    napari.run()


def track_single_dataset(
    dataset_entry: CTCDatasetEntry,
    sequence: str,
    model_entry: ONNXModelEntry,
    config: TrackingAccuracyConfig,
) -> dict:
    """Track one CTC sequence and evaluate metrics.

    Parameters
    ----------
    dataset_dir : Path
        CTC dataset root.
    sequence : str
        Sequence number (e.g. "01").
    model_entry : ONNXModelEntry
        Model to use (path=None for baseline).
    config : TrackingAccuracyConfig
        Evaluation configuration.

    Returns
    -------
    dict
        CTC metrics dict plus ``model``, ``dataset``, ``sequence`` keys.
    """
    dataset_dir = Path(dataset_entry.path)
    _seg_dir = seg_dir(dataset_dir, sequence)
    if not _seg_dir.exists():
        raise FileNotFoundError(f"Segmentation directory not found: {_seg_dir}")

    model_path = Path(model_entry.path) if model_entry.path is not None else None

    _logger.info("Loading labels from %s", _seg_dir)
    labels = imread(str(_seg_dir / "*.tif")).compute()
    images = imread(str(dataset_dir / sequence / "*.tif")).compute()

    gt_graph = td.graph.InMemoryGraph.from_ctc(dataset_dir / f"{sequence}_GT" / "TRA")

    _logger.info(
        "Tracking: model=%s dataset=%s seq=%s",
        model_entry.label,
        dataset_dir.name,
        sequence,
    )
    dataset_pixel_size = dataset_entry.pixel_size_um
    if dataset_pixel_size is None and config.ctc_metadata_path is not None:
        ctc_meta = _load_ctc_metadata(Path(config.ctc_metadata_path))
        dataset_pixel_size = ctc_meta.get(dataset_dir.name)
        if dataset_pixel_size is not None:
            _logger.info("Pixel size from metadata: %.4f µm/px (%s)", dataset_pixel_size, dataset_dir.name)
        else:
            _logger.warning(
                "Dataset %s not found in %s; no rescaling applied", dataset_dir.name, config.ctc_metadata_path
            )

    if model_entry.pixel_size_um is not None and dataset_pixel_size is not None:
        pixel_size_scale = dataset_pixel_size / model_entry.pixel_size_um
    else:
        pixel_size_scale = 1.0

    graph, solution_graph = _build_and_solve(model_path, images, labels, config, pixel_size_scale)

    if config.show_napari:
        _show_napari_viewer(graph, images, labels)

    _logger.info("Evaluating CTC metrics ...")
    metrics = td.metrics.evaluate_ctc_metrics(
        solution_graph,
        gt_graph,
        input_reset=False,
        reference_reset=False,
        metrics=config.ctc_metrics,
    )

    metrics["model"] = model_entry.label
    metrics["dataset"] = dataset_dir.name
    metrics["sequence"] = sequence
    return metrics


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to tracking accuracy YAML configuration file",
)
def main(config: Path) -> None:
    """Evaluate CTC tracking accuracy with DynaCLR ONNX embeddings.

    Runs ILP-based tracking on CTC benchmark datasets, comparing a spatial+IoU
    baseline against models that use DynaCLR embedding similarity as an additional
    edge cost. Writes results.csv to the configured output directory.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    raw = load_config(config)
    cfg = TrackingAccuracyConfig(**raw)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    for model_entry in cfg.models:
        for dataset_entry in cfg.datasets:
            dataset_dir = Path(dataset_entry.path)
            for sequence in dataset_entry.sequences:
                _seg = seg_dir(dataset_dir, sequence)
                if not _seg.exists():
                    click.echo(
                        f"Skipping {dataset_dir.name}/{sequence}: no segmentation at {_seg}",
                        err=True,
                    )
                    continue

                try:
                    row = track_single_dataset(dataset_entry, sequence, model_entry, cfg)
                except Exception as exc:
                    click.echo(
                        f"Error {model_entry.label} / {dataset_dir.name} / {sequence}: {exc}",
                        err=True,
                    )
                    _logger.exception("Tracking failed")
                    continue

                rprint(row)
                results.append(row)

                # Write incrementally so partial results are never lost
                df = pl.DataFrame(results)
                df.write_csv(output_dir / "results.csv")

    if not results:
        click.echo("No results produced.", err=True)
        return

    df = pl.DataFrame(results)
    df.write_csv(output_dir / "results.csv")
    click.echo(f"\nResults written to {output_dir / 'results.csv'}")

    # Summary: mean across sequences, grouped by model x dataset
    key_metrics = [c for c in ["LNK", "BIO(0)", "OP_CLB(0)", "CHOTA", "TRA", "DET"] if c in df.columns]
    if key_metrics:
        summary = df.group_by("model", "dataset").agg([pl.col(m).mean() for m in key_metrics]).sort("model", "dataset")
        click.echo("\n## Tracking Accuracy Summary (mean over sequences)\n")
        click.echo(summary.to_pandas().to_markdown(index=False, floatfmt=".3f"))


if __name__ == "__main__":
    main()
