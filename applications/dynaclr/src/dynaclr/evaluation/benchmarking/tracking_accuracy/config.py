"""Configuration models for CTC tracking accuracy evaluation."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ONNXModelEntry(BaseModel):
    """One model to benchmark.

    Parameters
    ----------
    path : str or None
        Path to the ONNX model file. None runs the baseline (IoU + spatial edges only,
        no embedding model).
    label : str
        Display name for this model in results.
    pixel_size_um : float or None
        Pixel size (µm/px) the model was trained at. Used to rescale input crops
        when the dataset pixel size differs. None disables rescaling.
    """

    path: str | None
    label: str
    pixel_size_um: float | None = None


class CTCDatasetEntry(BaseModel):
    """One CTC dataset directory.

    Parameters
    ----------
    path : str
        Path to the dataset root (e.g. /hpc/reference/group.royer/CTC/training/BF-C2DL-HSC).
        Must contain ``{seq}_ERR_SEG/``, ``{seq}/`` (raw images), and ``{seq}_GT/TRA/``
        subdirectories for each sequence.
    sequences : list[str]
        Sequence numbers to evaluate (e.g. ["01", "02"]).
    pixel_size_um : float or None
        Pixel size (µm/px) of the raw images. Used with ``ONNXModelEntry.pixel_size_um``
        to rescale crops before ONNX inference. If None, looked up from
        ``TrackingAccuracyConfig.ctc_metadata_path`` by dataset name, then
        falls back to reading TIFF XResolution metadata.
    """

    path: str
    sequences: list[str] = Field(default=["01", "02"])
    pixel_size_um: float | None = None


class TrackingAccuracyConfig(BaseModel):
    """Configuration for CTC tracking accuracy evaluation.

    Parameters
    ----------
    models : list[ONNXModelEntry]
        Models to benchmark. Include an entry with ``path: null`` for the IoU baseline.
    datasets : list[CTCDatasetEntry]
        CTC datasets to evaluate.
    model_input_shape : tuple[int, int]
        Height x width of the ONNX model input (must match what the model was exported with).
        Default (160, 160) matches the DynaCLR-2D-MIP training resolution.
    distance_threshold : float
        Maximum spatial distance (pixels) for candidate edges in DistanceEdges.
    n_neighbors : int
        Maximum candidate edges per cell.
    delta_t : int
        Maximum frame gap for candidate edges.
    division_weight : float
        ILP solver weight for cell division events.
    appearance_weight : float
        ILP solver weight for cell appearance.
    disappearance_weight : float
        ILP solver weight for cell disappearance.
    node_weight : float
        ILP solver weight per node (negative = prefer more detections).
    output_dir : str
        Directory for results CSV.
    ctc_metrics : list[str] or None
        CTC metric names to include in output. None = all available metrics.
    batch_size : int
        Number of cell crops per ONNX inference call.
    ctc_metadata_path : str or None
        Path to a CTC metadata YAML mapping dataset names to
        ``[interval_min, y_um, x_um]``. Used to look up pixel size when
        ``CTCDatasetEntry.pixel_size_um`` is not set. Falls back to reading
        TIFF XResolution tags if the dataset is not in the file.
    show_napari : bool
        Open a napari viewer after tracking each sequence. Only use when running
        interactively on a partition with a display. Default: False.
    """

    models: list[ONNXModelEntry] = Field(..., min_length=1)
    datasets: list[CTCDatasetEntry] = Field(..., min_length=1)
    ctc_metadata_path: str | None = None
    model_input_shape: tuple[int, int] = (160, 160)
    distance_threshold: float = 325.0
    n_neighbors: int = 10
    delta_t: int = 5
    division_weight: float = 0.5
    appearance_weight: float = 0.0
    disappearance_weight: float = 0.0
    node_weight: float = -10.0
    output_dir: str
    ctc_metrics: list[str] | None = None
    batch_size: int = 128
    show_napari: bool = False
