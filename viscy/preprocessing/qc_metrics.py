from abc import ABC, abstractmethod
from typing import Literal

import iohub.ngff as ngff
import numpy as np
from tqdm import tqdm

from viscy.utils.meta_utils import write_meta_field


class QCMetric(ABC):
    """Base class for composable QC metrics.

    Each metric:
    - Owns its channel list and per-channel config
    - Reads data and computes results per FOV
    - Returns structured dicts for zattrs storage
    """

    field_name: str

    @abstractmethod
    def channels(self) -> list[str] | Literal[-1]:
        """Channel names this metric operates on.

        Return -1 to operate on all channels in the dataset.
        """
        ...

    @abstractmethod
    def __call__(
        self,
        position: ngff.Position,
        channel_name: str,
        channel_index: int,
        num_workers: int = 4,
    ) -> dict:
        """Compute metric for one FOV and one channel.

        Returns
        -------
        dict
            {
                "fov_statistics": {"key": value, ...},
                "per_timepoint": {"0": value, "1": value, ...},
            }
        """
        ...


def generate_qc_metadata(
    zarr_dir: str,
    metrics: list[QCMetric],
    num_workers: int = 4,
) -> None:
    """Run composable QC metrics across an HCS dataset.

    Each metric specifies its own channels (or -1 for all).
    The orchestrator iterates positions, dispatches to each metric
    for its channels, aggregates dataset-level statistics, and
    writes to .zattrs.

    Parameters
    ----------
    zarr_dir : str
        Path to the HCS OME-Zarr dataset.
    metrics : list[QCMetric]
        List of QC metric instances to compute.
    num_workers : int
        Number of workers for data loading.
    """
    plate = ngff.open_ome_zarr(zarr_dir, mode="r+")
    position_map = list(plate.positions())

    for metric in metrics:
        channel_list = metric.channels()
        if channel_list == -1:
            channel_list = list(plate.channel_names)

        for channel_name in channel_list:
            channel_index = plate.channel_names.index(channel_name)
            print(f"Computing {metric.field_name} for channel '{channel_name}'")

            all_focus_values = []
            position_results = []

            for _, pos in tqdm(position_map, desc="Positions"):
                result = metric(pos, channel_name, channel_index, num_workers)
                position_results.append((pos, result))
                tp_values = list(result["per_timepoint"].values())
                all_focus_values.extend(tp_values)

            arr = np.array(all_focus_values, dtype=float)
            dataset_stats = {
                "z_focus_mean": float(np.mean(arr)),
                "z_focus_std": float(np.std(arr)),
                "z_focus_min": int(np.min(arr)),
                "z_focus_max": int(np.max(arr)),
            }

            write_meta_field(
                position=plate,
                metadata={"dataset_statistics": dataset_stats},
                field_name=metric.field_name,
                subfield_name=channel_name,
            )

            for pos, result in position_results:
                write_meta_field(
                    position=pos,
                    metadata={
                        "dataset_statistics": dataset_stats,
                        **result,
                    },
                    field_name=metric.field_name,
                    subfield_name=channel_name,
                )

    plate.close()
