"""Image 2D to vector / scalar conv net"""
from micro_dl.networks.base_image_to_vector_net import BaseImageToVectorNet


class Image2DToVectorNet(BaseImageToVectorNet):
    """Uses 2D images as input"""

    @property
    def _get_input_shape(self):
        """Return shape of input"""

        if self.config['data_format'] == 'channels_first':
            shape = (self.config['num_input_channels'],
                     self.config['height'],
                     self.config['width'])
        else:
            shape = (self.config['height'],
                     self.config['width'],
                     self.config['num_input_channels'])
        return shape
