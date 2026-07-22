"""Composable QC metrics for OME-Zarr datasets."""

import logging
from abc import ABC, abstractmethod

import iohub.ngff as ngff
from iohub.core.config import TensorStoreConfig
from tqdm import tqdm

from viscy_data.meta_csv import build_provenance_fields, metadata_to_rows, write_meta_rows_csv
from viscy_utils.meta_utils import write_meta_field

_logger = logging.getLogger(__name__)


class QCMetric(ABC):
    """Base class for composable QC metrics.

    Each metric:
    - Owns its channel list and per-channel config
    - Reads data and computes results per FOV
    - Returns structured dicts for zattrs storage
    """

    field_name: str

    @abstractmethod
    def channels(self) -> list[str]:
        """Channel names this metric operates on."""
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

    def aggregate_dataset(self, all_results: list[dict]) -> dict:
        """Compute dataset-level statistics from all position results.

        Parameters
        ----------
        all_results : list[dict]
            List of dicts returned by ``__call__`` for each position.

        Returns
        -------
        dict
            Dataset-level statistics to write under ``"dataset_statistics"``.
        """
        return {}


def generate_qc_metadata(
    zarr_dir: str,
    metrics: list[QCMetric],
    num_workers: int = 4,
    csv_dir: str | None = None,
) -> None:
    """Run composable QC metrics across an HCS dataset.

    Each metric specifies its own channels. The orchestrator iterates
    positions, dispatches to each metric for its channels, aggregates
    dataset-level statistics, and writes to .zattrs.

    Parameters
    ----------
    zarr_dir : str
        Path to the HCS OME-Zarr dataset.
    metrics : list[QCMetric]
        List of QC metric instances to compute.
    num_workers : int
        Number of workers for data loading.
    csv_dir : str or None, optional
        If given, results are written to a per-store CSV sidecar under
        this directory instead of into the store's ``.zattrs``, and the
        store is opened read-only. Use for datasets mounted without write
        access. By default None (write to ``.zattrs`` as usual).
    """
    with ngff.open_ome_zarr(
        zarr_dir,
        mode="r" if csv_dir else "r+",
        implementation="tensorstore",
        implementation_config=TensorStoreConfig(data_copy_concurrency=num_workers),
    ) as plate:
        position_map = list(plate.positions())

        for metric in metrics:
            channel_list = metric.channels()

            for channel_name in channel_list:
                channel_index = plate.channel_names.index(channel_name)
                _logger.info(f"Computing {metric.field_name} for channel '{channel_name}'")

                position_results = []

                for pos_name, pos in tqdm(position_map, desc="Positions"):
                    result = metric(pos, channel_name, channel_index, num_workers)
                    position_results.append((pos_name, pos, result))

                all_results = [r for _, _, r in position_results]
                dataset_stats = metric.aggregate_dataset(all_results)

                if csv_dir is not None:
                    provenance = build_provenance_fields()
                    rows = []
                    if dataset_stats:
                        rows.extend(
                            metadata_to_rows(
                                {"dataset_statistics": dataset_stats},
                                store_path=zarr_dir,
                                position_path=None,
                                channel_name=channel_name,
                                field_name=metric.field_name,
                            )
                        )
                    for pos_name, _, result in position_results:
                        metadata = {**result}
                        if dataset_stats:
                            metadata["dataset_statistics"] = dataset_stats
                        rows.extend(
                            metadata_to_rows(
                                metadata,
                                store_path=zarr_dir,
                                position_path=pos_name,
                                channel_name=channel_name,
                                field_name=metric.field_name,
                            )
                        )
                    for row in rows:
                        row.update(provenance)
                    write_meta_rows_csv(csv_dir, zarr_dir, rows)
                    continue

                if dataset_stats:
                    write_meta_field(
                        position=plate,
                        metadata={"dataset_statistics": dataset_stats},
                        field_name=metric.field_name,
                        subfield_name=channel_name,
                    )

                for _, pos, result in position_results:
                    metadata = {**result}
                    if dataset_stats:
                        metadata["dataset_statistics"] = dataset_stats
                    write_meta_field(
                        position=pos,
                        metadata=metadata,
                        field_name=metric.field_name,
                        subfield_name=channel_name,
                    )
