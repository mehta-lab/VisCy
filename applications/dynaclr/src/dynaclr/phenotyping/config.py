"""Explicit single-dataset recipe; collection expansion belongs to orchestration."""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Channel:
    """Describe a raw fluorescence channel and its marker in selected wells."""

    name: str
    marker: str
    role: str = "organelle"
    wells: tuple[str, ...] = ()  # empty selects every well


@dataclass(frozen=True)
class Annotation:
    """Identify an observation-only annotation store and its stated confidence."""

    path: str
    dataset: str
    confidence: str = "uncertain"


@dataclass(frozen=True)
class Segmentation:
    """Specify the strict CP-DINO model and optional working-grid spacing."""

    checkpoint: str = "cpdino"
    expected_backbone: str = "dino_vitl"
    target_pixel_size_um: float | None = None
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    min_size: int = 15  # pixels on the working grid, NOT native area
    batch_size: int = 8
    normalize_percentiles: tuple[float, float] = (1.0, 99.0)


@dataclass(frozen=True)
class Selection:
    """Describe FOV and frame selection for a staged dataset attempt."""

    fovs: tuple[str, ...] = ()
    times: tuple[int, ...] = ()  # empty means all frames
    annotation_enriched: bool = False


@dataclass(frozen=True)
class Config:
    """Collect the complete scientific settings for one dataset attempt."""

    dataset: str
    source: str
    focus_source: str
    tracks: str
    output: str
    membrane_channel: str = "membrane_prediction"
    nuclei_channel: str = "nuclei_prediction"
    focus_channel: str = "Phase3D"
    fluorescence: tuple[Channel, ...] = ()
    segmentation: Segmentation = field(default_factory=Segmentation)
    selection: Selection = field(default_factory=Selection)
    annotations: tuple[Annotation, ...] = ()
    exclude_fovs: tuple[str, ...] = ()
    track_label_mode: str = "auto"
    track_id_stride: int = 1_000_000
    phase_alignment_rtol: float = 0.0
    phase_alignment_atol: float = 0.0
    candidate_thresholds: dict[str, tuple[float, ...]] = field(default_factory=dict)
    max_overlays: int = 24

    def scientific_dict(self) -> dict[str, Any]:
        """Return scientific settings while allowing staged selection to expand."""
        result = asdict(self)
        for name in ("output", "selection", "max_overlays"):
            result.pop(name)
        return result


def _path(value: str, base: Path) -> str:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return str(path if path.is_absolute() else (base / path).resolve())


def normalize_fov(value: str) -> str:
    """Reject unsafe paths and retain zero padding (no guessed FOV remapping)."""
    result = value.strip("/")
    parts = result.split("/")
    if len(parts) != 3 or any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"Expected row/column/position FOV, got {value!r}")
    return result


def load_config(path: str | Path, dataset: str | None = None, output: str | Path | None = None) -> Config:
    """Load a single-dataset YAML recipe and reject ambiguous channel selectors."""
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Phenotyping config must be a mapping")
    raw = dict(raw)
    # Dataset override chooses an identity, not a different source behind its back.
    if dataset is not None and raw.get("dataset") != dataset:
        raise ValueError(f"Config dataset {raw.get('dataset')!r} does not match {dataset!r}")
    channels = raw.pop("channels", {})
    seg = dict(raw.pop("segmentation", {}))
    if "normalize_percentiles" in seg:
        seg["normalize_percentiles"] = tuple(seg["normalize_percentiles"])
    checkpoint = seg.get("checkpoint", "cpdino")
    if checkpoint != "cpdino":
        seg["checkpoint"] = _path(checkpoint, path.parent)
    selection = raw.pop("selection", {})
    annotations = raw.pop("annotations", [])
    for key in ("source", "focus_source", "tracks", "output"):
        if key not in raw:
            raise ValueError(f"Missing required config field: {key}")
        raw[key] = _path(raw[key], path.parent)
    if output is not None:
        raw["output"] = str(Path(output).resolve())
    raw["membrane_channel"] = channels.get("membrane", "membrane_prediction")
    raw["nuclei_channel"] = channels.get("nuclei", "nuclei_prediction")
    raw["focus_channel"] = channels.get("focus", "Phase3D")
    raw["fluorescence"] = tuple(
        Channel(**dict(ch, wells=tuple(ch.get("wells", ())))) for ch in channels.get("fluorescence", [])
    )
    raw["annotations"] = tuple(
        Annotation(_path(a["path"], path.parent), a["dataset"], a.get("confidence", "uncertain")) for a in annotations
    )
    raw["segmentation"] = Segmentation(**seg)
    raw["selection"] = Selection(
        tuple(normalize_fov(f) for f in selection.get("fovs", [])),
        tuple(int(t) for t in selection.get("times", [])),
        bool(selection.get("annotation_enriched", False)),
    )
    raw["exclude_fovs"] = tuple(normalize_fov(f) for f in raw.get("exclude_fovs", []))
    raw["candidate_thresholds"] = {
        k: tuple(float(x) for x in values) for k, values in raw.get("candidate_thresholds", {}).items()
    }
    config = Config(**raw)
    s = config.segmentation
    if s.expected_backbone != "dino_vitl":
        raise ValueError("This pipeline requires CP-DINO DINOv3 ViT-L (dino_vitl)")
    if s.target_pixel_size_um is not None and (
        not math.isfinite(s.target_pixel_size_um) or s.target_pixel_size_um <= 0
    ):
        raise ValueError("target_pixel_size_um must be null (native) or positive and finite")
    if not 0 <= s.normalize_percentiles[0] < s.normalize_percentiles[1] <= 100:
        raise ValueError("Invalid normalization percentiles")
    if s.min_size < 0 or s.batch_size < 1 or config.max_overlays < 0:
        raise ValueError("Invalid size/batch/overlay setting")
    if any(t < 0 for t in config.selection.times):
        raise ValueError("Frame indices must be nonnegative")
    if config.track_id_stride < 1 or config.track_label_mode not in (
        "auto",
        "track_id",
        "id",
        "id_modulo",
        "label_id",
        "segmentation_id",
    ):
        raise ValueError("Invalid track-label encoding")
    if any(ch.role not in ("organelle", "viral", "other") for ch in config.fluorescence):
        raise ValueError("Fluorescence role must be organelle, viral or other")
    markers = [ch.marker for ch in config.fluorescence]
    for ch in config.fluorescence:
        if any(len(w.split("/")) != 2 or any(x in ("", ".", "..") for x in w.split("/")) for w in ch.wells):
            raise ValueError("Channel wells must be exact row/column selectors")
    for i, first in enumerate(config.fluorescence):
        for second in config.fluorescence[i + 1 :]:
            overlap = not first.wells or not second.wells or bool(set(first.wells) & set(second.wells))
            if overlap and (first.name == second.name or first.marker == second.marker):
                raise ValueError("Duplicate fluorescence names/markers require disjoint well selectors")
    if config.phase_alignment_rtol < 0 or config.phase_alignment_atol < 0:
        raise ValueError("Alignment tolerances must be nonnegative")
    if any(not math.isfinite(x) for v in config.candidate_thresholds.values() for x in v):
        raise ValueError("Candidate thresholds must be finite")
    if set(config.candidate_thresholds) - set(markers):
        raise ValueError("Candidate thresholds reference an unknown fluorescence marker")
    return config


def channels_for_fov(config: Config, fov: str) -> tuple[Channel, ...]:
    """Resolve fluorescence markers using exact well selectors for a FOV."""
    well = "/".join(normalize_fov(fov).split("/")[:2])
    return tuple(ch for ch in config.fluorescence if not ch.wells or well in ch.wells)
