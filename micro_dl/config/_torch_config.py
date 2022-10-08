from typing import Set, Union, Literal
from pydantic import BaseModel, validator
from typing import Optional
import os
from torch.cuda import device_count

class _TorchModel(BaseModel):
    """
    Configuration validation skeleton for model subconfig
    of torch master config.
    """
    architecture: Literal['2.5D', '2D']
    in_channels: int
    out_channels: int
    residual: bool
    task: Literal['reg','seg']
    model_dir: str
    
    @validator('model_dir')
    def valid_path(cls, path):
        if not os.path.exists(path):
            Warning(f'Path specified in {str(cls)}  does not exist. Creating it '\
                'for you...')
            os.makedirs(path)
        return path

    @validator('in_channels', 'out_channels')
    def positive(cls, n):
        if not n > 0:
            raise ValueError(f'{validator.__name__} must be greater than 0.')
        return n


class _TorchTraining(BaseModel):
    """
    Configuration validation skeleton for model subconfig
    of torch master config.
    """
    epochs: int
    learning_rate: float
    optimizer: Literal['adam','sgd']
    loss: Literal['mse','l1','cossim','cel']
    testing_stride: int
    save_model_stride: int
    save_dir: str
    mask: bool
    mask_type: str
    device: Union[str, int]
    
    @validator('save_dir')
    def valid_path(cls, path):
        if not os.path.exists(path):
            Warning(f'Path specified in {str(cls)}  does not exist. Creating it '\
                'for you...')
            os.makedirs(path)
        return path

    @validator('epochs',
               'learning_rate',
               'testing_stride',
               'save_model_string',
               check_fields=False)
    def positive(cls, n):
        if not n > 0:
            raise ValueError(f'{validator.__name__} must be greater than 0.')
        return n
    
    @validator('device')
    def valid_device(cls, v):
        if isinstance(v, str):
            assert v == 'cpu' or v == 'gpu', "Device must be 'cpu','gpu' (for"\
                "autoselection), or a gpu index"
        elif isinstance(v, int):
            if v+1 > device_count():
                raise ValueError(f"gpu index {v} not in available gpus "\
                    f"{range(device_count())}")
        return v


class TorchConfig(BaseModel):
    """
    Configuration validation skeleton for torch master config.
    """
    preprocess_config_path : str
    train_config_path : str
    inference_config_path : str
    
    model: Union[_TorchModel, None]
    training: Union[_TorchTraining, None]
    
    @validator('preprocess_config_path',
               'train_config_path',
               'inference_config_path')
    def valid_path(cls, path):
        if not os.path.exists(path):
            Warning(f'Path specified in {str(cls)}  does not exist. Creating it'\
                ' for you...')
            os.makedirs(path)
        return path
