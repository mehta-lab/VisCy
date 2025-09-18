"""Generate masks from sum of flurophore channels."""

from pathlib import Path
from typing import Literal

import iohub.ngff as ngff
import viscy.utils.aux_utils as aux_utils
from viscy.utils.mp_utils import mp_create_and_write_mask


class MaskProcessor:
    """Appends Masks to zarr directories.

    Parameters
    ----------
    zarr_dir : Path
        Directory of HCS zarr store to pull data from. Note: data in store is assumed to be stored in TCZYX format.
    channel_ids : list[int] | int
        Channel indices to be masked (typically just one)
    time_ids : list[int] | int
        Timepoints to consider
    pos_ids : list[int] | int
        Position (FOV) indices to use
    num_workers : int, optional
        Number of workers for multiprocessing, by default 4
    mask_type : Literal["otsu", "unimodal", "mem_detection", "borders_weight_loss_map"], optional
        Method to use for generating mask. Needed for mapping to the masking function.
        One of: {'otsu', 'unimodal', 'mem_detection', 'borders_weight_loss_map'}, by default "otsu".
    overwrite_ok : bool, optional
        Overwrite existing masks, by default False.
    """

    def __init__(
        self,
        zarr_dir: Path,
        channel_ids: list[int] | int,
        time_ids: list[int] | int,
        pos_ids: list[int] | int,
        num_workers: int = 4,
        mask_type: Literal[
            "otsu", "unimodal", "mem_detection", "borders_weight_loss_map"
        ] = "otsu",
        overwrite_ok: bool = False,
    ):
        self.zarr_dir = zarr_dir
        self.num_workers = num_workers

        # Validate that given indices are available.
        metadata_ids = aux_utils.validate_metadata_indices(
            zarr_dir=zarr_dir,
            time_ids=time_ids,
            channel_ids=channel_ids,
            pos_ids=pos_ids,
        )
        self.time_ids = metadata_ids["time_ids"]
        self.channel_ids = metadata_ids["channel_ids"]
        self.position_ids = metadata_ids["pos_ids"]

        assert mask_type in [
            "otsu",
            "unimodal",
            "mem_detection",
            "borders_weight_loss_map",
        ], (
            "Masking method invalid, 'otsu', 'unimodal', 'mem_detection', "
            "'borders_weight_loss_map' are supported"
        )
        self.mask_type = mask_type
        self.ints_metadata = None
        self.channel_thr_df = None

        plate = ngff.open_ome_zarr(store_path=zarr_dir, mode="r")

        # deal with output channel selection/overwriting messages
        if overwrite_ok:
            mask_name = "_".join(["mask", self.mask_type])
            if mask_name in plate.channel_names:
                print(f"Mask found in channel {mask_name}. Overwriting with this mask.")
        plate.close()

    def generate_masks(self, structure_elem_radius: int = 5):
        """Generate foreground masks from fluorophore channels.

        The sum of flurophore channels is thresholded to generate a foreground
        mask.

        Masks are saved as an additional channel in each data array for each
        specified position. If certain channels are not specified, gaps are
        filled with arrays of zeros.

        Masks are also saved as an additional untracked array named "mask" and
        tracked in the "mask" metadata field.

        Parameters
        ----------
        structure_elem_radius : int
            Radius of structuring element for morphological operations
        """
        # Gather function arguments for each index pair at each position
        plate = ngff.open_ome_zarr(store_path=self.zarr_dir, mode="r+")

        mp_mask_creator_args = []

        for i, (_, position) in enumerate(plate.positions()):
            # TODO: make a better progress bar for mask generation
            verbose = i % 4 == 0
            mp_mask_creator_args.append(
                tuple(
                    [
                        position,
                        self.time_ids,
                        self.channel_ids,
                        structure_elem_radius,
                        self.mask_type,
                        "_".join(["mask", self.mask_type]),
                        verbose,
                    ]
                )
            )

        # create and write masks and metadata using multiprocessing
        mp_create_and_write_mask(mp_mask_creator_args, workers=self.num_workers)

        plate.close()
