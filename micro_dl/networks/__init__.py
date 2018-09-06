"""Classes related to different NN architectures"""
from .image2D_to_vector_net import Image2DToVectorNet
from .image3D_to_vector_net import Image3DToVectorNet
from .layers.interp_upsampling2D import InterpUpSampling2D
from .layers.interp_upsampling3D import InterpUpSampling3D
from .unet2D import UNet2D
from .unet3D import UNet3D
from .unet_stack_2D import UNetStackTo2D
from .unet_stack_stack import UNetStackToStack
