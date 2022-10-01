import numpy as np
import torch

import micro_dl.inference.model_inference as inference
import micro_dl.torch_unet.utils.model as model_utils
import micro_dl.torch_unet.utils.dataset as ds

class TorchPredictor():
    '''
    Instance of TorchPredictor which performs all of the compatibility functions required to
    run prediction on an image given a trained pytorch model
    
    Params:
        - model -> nn.module: trained model using gpu parameters
        - network_config -> dict: model configuration dictionary. parameters can be found in torch_unet.utils.readme.md
        - device -> torch.device: device to run inference on
    '''
    def __init__(self, model = None, network_config = None, device = None) -> None:
        self.model = model
        self.network_config = network_config
        self.device = device
        
    def load_model_torch(self) -> None:
        '''
        Initializes a model according to the network configuration dictionary used to train it, and loads the 
        parameters saved in model_dir into the model's state dict.
        '''
        model = model_utils.model_init(self.network_config, device=self.device)
        
        model_dir = self.network_config['model_dir']
        readout = model.load_state_dict(torch.load(model_dir))
        print(f'PyTorch model load status: {readout}')
        self.model = model
    
    def predict_image(self, input_image, model = None):
        '''
        Alias for predict_large_image if 2.5D; 2.5D torch model is xy generalizable
        '''
        return self.predict_large_image(input_image, model=model)

    def predict_large_image(self, input_image, model = None):
        '''
        Runs prediction on entire image field of view. xy size is configurable, but it must be
        a power of 2.
        
        Params:
            - input_image -> numpy.ndarray or torch.Tensor: input image or image stack on which to run prediction
            - model -> Torch.nn.Module: trained model to use for prediction
        '''
        assert self.model != None or model != None, 'model must be specified in initiation or prediction call'
        if model == None:
            model = self.model
            
        if self.network_config['architecture'] == '2.5D':
            img_tensor = ds.ToTensor(device=self.device)(input_image)
            pred = model(img_tensor)
        elif self.network_config['architecture'] == '2D':
            img_tensor = ds.ToTensor(device=self.device)(input_image)[...,0,:,:]
            pred = torch.unsqueeze(model(img_tensor), -3)
            
        return pred.detach().cpu().numpy()

    def predict_large_image_tiling(self, input_image, model = None):
        '''
        Takes large image (or image stack) as a numpy array and returns prediction for it, block-wise.
        
        Note: only accepts input_image inputs of sizes that are powers of 2 (due to downsampling). Please crop 
        non-power-of-two length images.
        
        Params:
            - input_image -> numpy.ndarray or torch.Tensor: large (xy > 256x256) input image or image stack
            - model -> Torch.nn.Module: trained model to use for prediction
        
        '''
        assert len(input_image.shape) in [4, 5],'Invalid image shape: only 4D and 5D inputs - 2D / 3D images with channel and batch dim allowed'
        
        if type(input_image) != type(torch.rand(1,1)):
            input_image = ds.ToTensor(device=self.device)(input_image)
        
        #generate tiles
        tiles = self.tile_large_image_torch(input_image)
        #predict each tile
        for key in tiles:
            for ind, input_tile in enumerate(tiles[key]):
                prediction = model(input_tile)
                
                tiles[key][ind] = prediction
        #stitich predictions
        output_image = self.stitch_image_tiles_torch(tiles, (input_image.shape[-2], input_image.shape[-1]))
        
        return output_image.detach().cpu().numpy()

    def tile_large_image(self, input_image, tile_size = (256,256), stride = 128):
        #TODO: implement tiling with a stride.
        '''
        Takes large input image as and returns dictionary of tiles of that image.
        dictionaries are indexed by row number of each tile (tile-sized rows),
        and contains list of all tiles of that row.
        
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
        
    def stitch_image_tiles(self, tiles, output_size, stride = 128):
        #TODO: Implement stride and stitching using center of FoV only
        #      Offer option for averaging versus completely overriding overlap
        '''
        Takes in dictionary of tiles in,
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