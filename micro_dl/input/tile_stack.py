"""Tile images for training"""

import numpy as np
import os
import pandas as pd

import micro_dl.utils.aux_utils as aux_utils
from micro_dl.utils.normalize import hist_clipping, zscore
from micro_dl.utils.aux_utils import get_meta_idx
import micro_dl.utils.image_utils as image_utils


class ImageStackTiler:
    """Tiles all images images in a stack"""

    def __init__(self,
                 input_dir,
                 output_dir,
                 tile_size,
                 step_size,
                 time_ids=-1,
                 channel_ids=-1,
                 slice_ids=-1,
                 hist_clip_limits=None,
                 flat_field_dir=None,
                 isotropic=False):
        """
        Normalizes images using z-score, then tiles them.
        Isotropic here refers to the same dimension/shape along row, col, slice
        and not really isotropic resolution in mm.

        :param str input_dir: Directory with frames to be tiled
        :param str output_dir: Directory for storing the tiled images
        :param list/tuple/np array tile_size: size of the blocks to be cropped
         from the image
        :param list/tuple/np array step_size: size of the window shift. In case
         of no overlap, the step size is tile_size. If overlap, step_size <
         tile_size
        :param list/int time_ids: Tile given timepoint indices
        :param list/int tile_channels: Tile images in the given channel indices
         default=-1, tile all channels
        :param int slice_ids: Index of which focal plane acquisition to
         use (for 2D). default=-1 for the whole z-stack
        :param list hist_clip_limits: lower and upper percentiles used for
         histogram clipping.
        :param str flat_field_dir: Flatfield directory. None if no flatfield
            correction
        :param bool isotropic: if 3D, make the grid/shape isotropic
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.flat_field_dir = flat_field_dir
        self.frames_metadata = aux_utils.read_meta(self.input_dir)
        # Get metadata indices
        metadata_ids = aux_utils.validate_metadata_indices(
            frames_metadata=self.frames_metadata,
            time_ids=time_ids,
            channel_ids=channel_ids,
            slice_ids=slice_ids,
        )
        self.channels_ids = metadata_ids['channel_ids']
        self.time_ids = metadata_ids['time_ids']
        self.slice_ids = metadata_ids['slice_ids']

        self.tile_size = tile_size
        self.step_size = step_size
        self.isotropic = isotropic
        self.hist_clip_limits = hist_clip_limits

    def _preprocess_im(self,
                       time_idx,
                       channel_idx,
                       slice_idx,
                       pos_idx,
                       flat_field_im=None,
                       hist_clip_limits=None):
        """
        Preprocess image given by indices: flatfield correction, histogram
        clipping and z-score normalization is performed.

        :param int time_idx: Time index
        :param int channel_idx: Channel index
        :param int slice_idx: Slice (z) index
        :param int pos_idx: Position (FOV) index
        :param np.array flat_field_im: Flat field image for channel
        :param list hist_clip_limits: Limits for histogram clipping (size 2)
        :return np.array im: 2D preprocessed image
        :return str channel_name: Channel name
        """
        meta_idx = get_meta_idx(
            self.frames_metadata,
            time_idx,
            channel_idx,
            slice_idx,
            pos_idx,
        )
        channel_name = self.frames_metadata.loc[meta_idx, "channel_name"]
        file_path = os.path.join(
            self.input_dir,
            self.frames_metadata.loc[meta_idx, "file_name"],
        )
        im = image_utils.read_image(file_path)
        if flat_field_im is not None:
            im = image_utils.apply_flat_field_correction(
                im,
                flat_field_image=flat_field_im,
            )
        # normalize
        if hist_clip_limits is not None:
            im = hist_clipping(
                im,
                hist_clip_limits[0],
                hist_clip_limits[1],
            )
        return zscore(im), channel_name

    def _write_tiled_data(self,
                          tiled_data,
                          save_dir,
                          time_idx=None,
                          channel_idx=None,
                          slice_idx=None,
                          pos_idx=None,
                          tile_indices=None,
                          tiled_metadata=None,
                          ):
        """
        Loops through tuple and writes all tile image data. Adds row to metadata
        dataframe as well if that is present.

        :param list of tuples tiled_data: Tile name and np.array
        :param str save_dir: Directory where tiles will be written
        :param int time_idx: Time index
        :param int channel_idx: Channel index
        :param int slice_idx: Slice (z) index
        :param int pos_idx: Position (FOV) index
        :param list of tuples tile_indices: Tile indices
        :param dataframe tiled_metadata: Dataframe containing metadata for all
         tiles
        :return dataframe tiled_metadata: Metadata with rows added to it
        """
        for i, data_tuple in enumerate(tiled_data):
            rcsl_idx = data_tuple[0]
            file_name = aux_utils.get_im_name(
                time_idx=time_idx,
                channel_idx=channel_idx,
                slice_idx=slice_idx,
                pos_idx=pos_idx,
                extra_field=rcsl_idx,
            )
            np.save(os.path.join(save_dir, file_name),
                    data_tuple[1],
                    allow_pickle=True,
                    fix_imports=True)
            tile_idx = tile_indices[i]
            if tiled_metadata is not None:
                tiled_metadata = tiled_metadata.append(
                    {"channel_idx": channel_idx,
                     "slice_idx": slice_idx,
                     "time_idx": time_idx,
                     "file_name": file_name,
                     "pos_idx": pos_idx,
                     "row_start": tile_idx[0],
                     "col_start": tile_idx[2],
                     },
                    ignore_index=True,
                )
        return tiled_metadata

    def _get_flat_field(self, channel_idx):
        """
        Get flat field image for a given channel index

        :param int channel_idx: Channel index
        :return np.array flat_field_im: flat field image for channel
        """
        flat_field_im = None
        if self.flat_field_dir is not None:
            flat_field_im = np.load(
                os.path.join(
                    self.flat_field_dir,
                    'flat-field_channel-{}.npy'.format(channel_idx),
                )
            )
        return flat_field_im

    def _get_dataframe(self):
        """
        Creates an empty dataframe with metadata column names for tiles. It's
        the same names as for frames, but with channel_name removed and with
        the addition of row_start and col_start.
        TODO: Should I also save row_end and col_end while I'm at it?
        Might be useful if we want to recreate tiles from a previous preprocessing
        with mask run... Or just retrieve tile_size from preprocessing_info...
        This is one of the functions that will have to be adapted once tested on
        3D data.

        :return dataframe tiled_metadata
        """
        return pd.DataFrame(columns=[
            "channel_idx",
            "slice_idx",
            "time_idx",
            "file_name",
            "pos_idx",
            "row_start",
            "col_start"])

    def tile_stack(self):
        """
        Tiles images in the specified channels.

        Saves a csv with columns
        ['time_idx', 'channel_idx', 'pos_idx','slice_idx', 'file_name']
        for all the tiles
        """
        tiled_metadata = self._get_dataframe()
        tile_indices = None
        for channel_idx in self.channels_ids:
            # Perform flatfield correction if flatfield dir is specified
            flat_field_im = self._get_flat_field(channel_idx=channel_idx)

            for slice_idx in self.slice_ids:
                for time_idx in self.time_ids:
                    for pos_idx in np.unique(self.frames_metadata["pos_idx"]):
                        im, channel_name = self._preprocess_im(
                            time_idx,
                            channel_idx,
                            slice_idx,
                            pos_idx,
                            flat_field_im=flat_field_im,
                            hist_clip_limits=self.hist_clip_limits,
                        )
                        # Now to the actual tiling
                        if tile_indices is None:
                            tiled_image_data, tile_indices =\
                                image_utils.tile_image(
                                    input_image=im,
                                    tile_size=self.tile_size,
                                    step_size=self.step_size,
                                    isotropic=self.isotropic,
                                )
                        else:
                            tiled_image_data = image_utils.crop_at_indices(
                                input_image=im,
                                crop_indices=tile_indices,
                                isotropic=self.isotropic,
                            )
                        tiled_metadata = self._write_tiled_data(
                            tiled_image_data,
                            save_dir=self.output_dir,
                            time_idx=time_idx,
                            channel_idx=channel_idx,
                            slice_idx=slice_idx,
                            pos_idx=pos_idx,
                            tile_indices=tile_indices,
                            tiled_metadata=tiled_metadata,
                        )
        # Finally, save all the metadata
        tiled_metadata.to_csv(
            os.path.join(self.output_dir, "frames_meta.csv"),
            sep=",",
        )

    def tile_mask_stack(self,
                        mask_dir=None,
                        tile_mask_dir=None,
                        min_fraction=None,
                        isotropic=False):
        """
        Tiles images in the specified channels assuming there are masks
        already created in mask_dir. Only tiles above a certain fraction
        of foreground in mask tile will be saved and added to metadata.

        Saves a csv with columns ['time_idx', 'channel_idx', 'pos_idx',
        'slice_idx', 'file_name'] for all the tiles

        :param str mask_dir: Directory containing masks
        :param str tile_mask_dir: Directory where tiled masks will be saved
        :param float min_fraction: Minimum fraction of foreground in tiled masks
        :param bool isotropic: Indicator of isotropy
        """
        tiled_metadata = self._get_dataframe()
        # Load flatfield images if flatfield dir is specified
        flat_field_im = None
        if self.flat_field_dir is not None:
            flat_field_ims = []
            for channel_idx in self.channels_ids:
                flat_field_ims.append(self._get_flat_field(channel_idx))

        for slice_idx in self.slice_ids:
            for time_idx in self.time_ids:
                for pos_idx in np.unique(self.frames_metadata["pos_idx"]):
                    # Since masks are generated across channels, we only need
                    # load them once across channels (masks have no channel info
                    # in file name)
                    file_name = aux_utils.get_im_name(
                        time_idx=time_idx,
                        slice_idx=slice_idx,
                        pos_idx=pos_idx,
                    )
                    file_path = os.path.join(
                        mask_dir,
                        file_name,
                    )
                    mask_image = image_utils.read_image(file_path)
                    tiled_mask_data, tile_indices = image_utils.tile_image(
                        input_image=mask_image,
                        min_fraction=min_fraction,
                        tile_size=self.tile_size,
                        step_size=self.step_size,
                        isotropic=isotropic,
                        return_index=True,
                    )
                    # Loop through all the mask tiles, write tiled masks
                    mask_metadata = self._get_dataframe()
                    mask_metadata = self._write_tiled_data(
                        tiled_data=tiled_mask_data,
                        save_dir=tile_mask_dir,
                        time_idx=time_idx,
                        slice_idx=slice_idx,
                        pos_idx=pos_idx,
                        tile_indices=tile_indices,
                        tiled_metadata=mask_metadata,
                    )
                    mask_metadata.to_csv(
                        os.path.join(tile_mask_dir, "frames_meta.csv"),
                        sep=",",
                    )
                    # Loop through all channels and tile from indices
                    for i, channel_idx in enumerate(self.channels_ids):
                        if self.flat_field_dir is not None:
                            flat_field_im = flat_field_ims[i]

                        im, channel_name = self._preprocess_im(
                            time_idx,
                            channel_idx,
                            slice_idx,
                            pos_idx,
                            flat_field_im=flat_field_im,
                            hist_clip_limits=self.hist_clip_limits,
                        )
                        # Now to the actual tiling
                        tiled_image_data = image_utils.crop_at_indices(
                            input_image=im,
                            crop_indices=tile_indices,
                            isotropic=self.isotropic,
                        )
                        # Loop through all the tiles, write and add to metadata
                        tiled_metadata = self._write_tiled_data(
                            tiled_data=tiled_image_data,
                            save_dir=self.output_dir,
                            time_idx=time_idx,
                            channel_idx=channel_idx,
                            slice_idx=slice_idx,
                            pos_idx=pos_idx,
                            tile_indices=tile_indices,
                            tiled_metadata=tiled_metadata,
                        )

        # Finally, save all the metadata
        tiled_metadata.to_csv(
            os.path.join(self.output_dir, "frames_meta.csv"),
            sep=",",
        )
