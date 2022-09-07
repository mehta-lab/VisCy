import numpy as np
import torch

import micro_dl.inference.model_inference as inference
import micro_dl.torch_unet.utils.model as model_utils
import micro_dl.torch_unet.utils.dataset as ds


def load_model_torch(model_dir, network_config):
    '''
    Initializes a model according to the network configuration dictionary used to train it, and loads the 
    parameters saved in model_dir into the model's state dict.
    
    Params:
        - model_dir -> str: directory containing the model parameters (the dir used in model.save_dict(*here*))
        - network_config -> dict: model configuration dictionary
    '''
    model = model_utils.model_init(network_config)
    model.load_state_dict(torch.load(model_dir))
    return model



def predict_large_image_torch(model, input_image):
    '''
    Takes large image (or image stack) as a numpy array and returns prediction for it, block-wise.
    
    Note: only accepts input_image inputs of sizes that are powers of 2 (due to downsampling). Please crop 
    non-power-of-two length images.
    
    Params:
        - model -> Torch.nn.Module: trained model to use for prediction
        - input_image -> numpy.ndarray or torch.Tensor: large (>256 x 256) input image
    
    '''
    assert len(input_image.shape) in [4, 5],''.join('Invalid image shape: only 4D and 5D inputs - 2D / 3D',
                                                    'images with channel and batch dim allowed')
    if type(input_image) != type(torch.rand(1,1)):
        norm_value = np.max(input_image)
        input_image = ds.ToTensor()(input_image)
    else:
        norm_value = torch.max(input_image)
    
    tiles = tile_large_image_torch(input_image)
    
    for key in tiles:
        for ind, input_tile in enumerate(tiles[key]):
            prediction = model(input_tile/norm_value)
            
            tiles[key][ind] = prediction
    
    output_image = stitch_image_tiles_torch(tiles, (input_image.shape[-2], input_image.shape[-1]))
    
    return output_image.detach().cpu().numpy()



def tile_large_image_torch(input_image, tile_size = (256,256)):
    '''
    Takes large input image as and returns dictionary of tiles of that image
    
    Params:
        input_image
    '''
    img_shape = input_image.shape
    
    assert img_shape[0] % tile_size[0] and img_shape[1] % tile_size[1], ''.join('For downsampling reasons, image input size',
                                                                                'must be multiple of tile input size')
    
    img_dict = {}
    for i in range(img_shape[-2]//tile_size[0]):
        row = []
        for j in range(img_shape[-1]//tile_size[1]):
            img_tile = input_image[...,i*tile_size[0]:(i+1)*tile_size[0], j*tile_size[0]:(j+1)*tile_size[0]]
            row.append(img_tile)
        img_dict[i] = row
    return img_dict
            

    
def stitch_image_tiles_torch(tiles, output_size):
    '''
    Takes in dictionary of tiles in ,
    and returns the tiles stitched together along x and y dimensions.
    
    Params:
        tiles -> dict: format {row_index: [row of torch.tensor tiles], ..., row_index: [row of torch.tensor tiles]} 
        output_size -> tuple (int, int): expected x-y size of output tensor
    '''
    
    output_tensor = []
    for key in tiles:
        output_tensor.append(tiles[key])
        
    output_tensor = [torch.cat(row, axis = -1) for row in output_tensor]
    output_tensor = torch.cat(output_tensor, axis = -2)
    
    return output_tensor