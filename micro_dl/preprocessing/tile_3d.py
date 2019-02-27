"""Tile 3D"""
from micro_dl.preprocessing.tile_uniform_images import ImageTilerUniform


class ImageTilerUniform3D(ImageTilerUniform):
    """Tiles all volumes in a dataset"""

    def __init__(self,
                 input_dir,
                 output_dir,
                 tile_dict,
                 tile_size=[64, 64, 64],
                 step_size=[32, 32, 32],
                 depths=1,
                 time_ids=-1,
                 channel_ids=-1,
                 slice_ids=-1,
                 pos_ids=-1,
                 hist_clip_limits=None,
                 flat_field_dir=None,
                 image_format='zyx',
                 num_workers=4,
                 int2str_len=3):
        """Init

        Please ref to init of ImageTilerUniform.
        Assuming slice_ids are contiguous
        Depth=1 here. slice_idx is used to store slice_start_idx.

        """

        super().__init__(input_dir,
                         output_dir,
                         tile_dict,
                         tile_size,
                         step_size,
                         depths,
                         time_ids,
                         channel_ids,
                         slice_ids,
                         pos_ids,
                         hist_clip_limits,
                         flat_field_dir,
                         image_format,
                         num_workers,
                         int2str_len)

        if isinstance(self.tile_size, list):
            assert len(self.tile_size) == 3, \
                'tile size missing for some dimensions'

        if isinstance(self.step_size, list):
            assert len(self.step_size) == 3, \
                'step size missing for some dimensions'

        assert self.tile_3d is True
        assert all([item == 1 for item in self.channel_depth.values()]), \
            'Depth is > 1 for 3D volume. ' \
            'Tiling does not support 4D arrays currently'
