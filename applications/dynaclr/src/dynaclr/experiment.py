"""Experiment configuration and registry for multi-experiment DynaCLR training.

Provides :class:`ExperimentConfig` (per-experiment metadata) and
:class:`ExperimentRegistry` (validated collection with channel resolution,
YAML loading, and tau-range conversion).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from iohub.ngff import open_ome_zarr

_logger = logging.getLogger(__name__)

__all__ = ["ExperimentConfig", "ExperimentRegistry"]


# ---------------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment in a multi-experiment training setup.

    Parameters
    ----------
    name : str
        Unique identifier for this experiment.
    data_path : str
        Path to the HCS OME-Zarr store.
    tracks_path : str
        Root directory for per-FOV tracking CSVs.
    channel_names : list[str]
        All channel names present in the zarr store.
    source_channel : list[str]
        Which channels to use for training (subset of *channel_names*).
    condition_wells : dict[str, list[str]]
        Mapping of condition label to well names (e.g. ``{"uninfected": ["A/1"]}``).
    interval_minutes : float
        Time between frames in minutes.
    start_hpi : float
        Hours post infection (or perturbation) at frame 0.
    organelle : str
        Optional organelle label.
    date : str
        Optional experiment date string.
    moi : float
        Multiplicity of infection (0.0 if not applicable).
    """

    name: str
    data_path: str
    tracks_path: str
    channel_names: list[str]
    source_channel: list[str]
    condition_wells: dict[str, list[str]]
    interval_minutes: float = 30.0
    start_hpi: float = 0.0
    organelle: str = ""
    date: str = ""
    moi: float = 0.0


# ---------------------------------------------------------------------------
# ExperimentRegistry
# ---------------------------------------------------------------------------


@dataclass
class ExperimentRegistry:
    """Validated collection of :class:`ExperimentConfig` instances.

    On creation (``__post_init__``), the registry performs fail-fast validation:

    1. Experiments list must not be empty.
    2. Experiment names must be unique.
    3. Each experiment's ``source_channel`` entries must exist in its ``channel_names``.
    4. All experiments must have the same number of ``source_channel`` entries.
    5. ``interval_minutes`` must be positive for each experiment.
    6. ``condition_wells`` must not be empty for each experiment.
    7. ``data_path`` must point to an existing directory.
    8. Zarr metadata channel names must match ``channel_names``.

    After validation the registry computes:

    * ``num_source_channels`` -- common count of source channels.
    * ``channel_maps`` -- per-experiment mapping of source position to zarr
      channel index.

    Parameters
    ----------
    experiments : list[ExperimentConfig]
        List of experiment configurations.
    """

    experiments: list[ExperimentConfig]
    num_source_channels: int = field(init=False)
    channel_maps: dict[str, dict[int, int]] = field(init=False)

    # internal lookup
    _name_map: dict[str, ExperimentConfig] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        # 1. Empty check
        if not self.experiments:
            raise ValueError(
                "Empty experiments list: at least one ExperimentConfig is required."
            )

        # 2. Duplicate names
        names: list[str] = [e.name for e in self.experiments]
        seen: set[str] = set()
        for n in names:
            if n in seen:
                raise ValueError(
                    f"Duplicate experiment name '{n}'. "
                    "Each experiment must have a unique name."
                )
            seen.add(n)

        # Build name -> config map
        self._name_map = {e.name: e for e in self.experiments}

        # Per-experiment validations
        for exp in self.experiments:
            # 5. Negative interval
            if exp.interval_minutes <= 0:
                raise ValueError(
                    f"Experiment '{exp.name}': interval_minutes must be "
                    f"positive, got {exp.interval_minutes}."
                )

            # 6. Empty condition_wells
            if not exp.condition_wells:
                raise ValueError(
                    f"Experiment '{exp.name}': condition_wells must not be empty."
                )

            # 3. Source channel membership
            missing = [
                ch for ch in exp.source_channel if ch not in exp.channel_names
            ]
            if missing:
                raise ValueError(
                    f"Experiment '{exp.name}': source_channel entries "
                    f"{missing} not found in channel_names {exp.channel_names}."
                )

            # 7. data_path existence
            if not Path(exp.data_path).exists():
                raise ValueError(
                    f"Experiment '{exp.name}': data_path does not exist: "
                    f"{exp.data_path}"
                )

            # 8. Zarr channel validation
            with open_ome_zarr(exp.data_path, mode="r") as plate:
                first_position = next(iter(plate.positions()))[1]
                zarr_channels = list(first_position.channel_names)
            if zarr_channels != exp.channel_names:
                raise ValueError(
                    f"Experiment '{exp.name}': channel_names mismatch. "
                    f"Expected (from config): {exp.channel_names}, "
                    f"got (from zarr): {zarr_channels}."
                )

        # 4. Consistent source channel count
        counts = {len(e.source_channel) for e in self.experiments}
        if len(counts) > 1:
            detail = ", ".join(
                f"'{e.name}': {len(e.source_channel)}"
                for e in self.experiments
            )
            raise ValueError(
                f"All experiments must have the same number of source_channel "
                f"entries, but found: {detail}."
            )
        self.num_source_channels = counts.pop()

        # Compute channel_maps
        self.channel_maps = {}
        for exp in self.experiments:
            self.channel_maps[exp.name] = {
                i: exp.channel_names.index(sc)
                for i, sc in enumerate(exp.source_channel)
            }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentRegistry:
        """Load experiments from a YAML file.

        Expected YAML structure::

            experiments:
              - name: "exp_a"
                data_path: "/path/to/exp_a.zarr"
                tracks_path: "/path/to/tracks"
                channel_names: ["Phase", "GFP"]
                source_channel: ["Phase"]
                condition_wells:
                  uninfected: ["A/1"]
                interval_minutes: 30.0

        Parameters
        ----------
        path : str | Path
            Path to the YAML configuration file.

        Returns
        -------
        ExperimentRegistry
            Validated registry of experiments.
        """
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        configs = [ExperimentConfig(**entry) for entry in data["experiments"]]
        return cls(experiments=configs)

    def tau_range_frames(
        self,
        experiment_name: str,
        tau_range_hours: tuple[float, float],
    ) -> tuple[int, int]:
        """Convert a tau range from hours to frames for a given experiment.

        Parameters
        ----------
        experiment_name : str
            Name of the experiment whose ``interval_minutes`` is used.
        tau_range_hours : tuple[float, float]
            ``(min_hours, max_hours)`` range.

        Returns
        -------
        tuple[int, int]
            ``(min_frames, max_frames)`` after conversion.
        """
        exp = self.get_experiment(experiment_name)
        min_frames = round(tau_range_hours[0] * 60 / exp.interval_minutes)
        max_frames = round(tau_range_hours[1] * 60 / exp.interval_minutes)

        if min_frames >= max_frames:
            _logger.warning(
                "Experiment '%s': tau_range_hours=%s yields fewer than 2 "
                "valid frames (min=%d, max=%d).",
                experiment_name,
                tau_range_hours,
                min_frames,
                max_frames,
            )

        return (min_frames, max_frames)

    def get_experiment(self, name: str) -> ExperimentConfig:
        """Look up an experiment by name.

        Parameters
        ----------
        name : str
            Experiment name.

        Returns
        -------
        ExperimentConfig

        Raises
        ------
        KeyError
            If *name* is not in the registry.
        """
        try:
            return self._name_map[name]
        except KeyError:
            raise KeyError(
                f"Experiment '{name}' not found in registry. "
                f"Available: {list(self._name_map.keys())}"
            )
