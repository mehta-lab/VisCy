"""Generate masks from sum of flurophore channels"""

import iohub.ngff as ngff

import viscy.utils.aux_utils as aux_utils
from viscy.utils.mp_utils import mp_create_and_write_mask


class MaskProcessor:
    """
    Appends Masks to zarr directories
    """

    def __init__(
        self,
        zarr_dir,
        channel_ids,
        time_ids=-1,
        pos_ids=-1,
        num_workers=4,
        mask_type="otsu",
        overwrite_ok=False,
    ):
        """
        :param str zarr_dir: directory of HCS zarr store to pull data from.
                            Note: data in store is assumed to be stored in
                            (time, channel, z, y, x) format.
        :param list[int] channel_ids: Channel indices to be masked (typically
            just one)
        :param int/list channel_ids: generate mask from the sum of these
            (flurophore) channel indices
        :param list/int time_ids: timepoints to consider
        :param int pos_ids: Position (FOV) indices to use
        :param int num_workers: number of workers for multiprocessing
        :param str mask_type: method to use for generating mask. Needed for
            mapping to the masking function. One of:
                {'otsu', 'unimodal', 'borders_weight_loss_map'}
        """
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

    def generate_masks(self, structure_elem_radius=5):
        """
        The sum of flurophore channels is thresholded to generate a foreground
        mask.

        Masks are saved as an additional channel in each data array for each
        specified position. If certain channels are not specified, gaps are
        filled with arrays of zeros.

        Masks are also saved as an additional untracked array named "mask" and
        tracked in the "mask" metadata field.

        :param int structure_elem_radius: Radius of structuring element for
                                morphological operations
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
