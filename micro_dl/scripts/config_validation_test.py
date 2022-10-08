from micro_dl.config._torch_config import TorchConfig
import micro_dl.utils.aux_utils as aux_utils
import os

kidney_25D_config = os.path.join('/hpc/projects/compmicro/projects',
                                'virtualstaining/torch_microDL/config_files',
                                '2019_02_15_KidneyTissue_DLMBL_subset/09_30_2022_12_06/',
                                'torch_config_25D.yml')
config_data = aux_utils.read_config(kidney_25D_config)

#try invalid input it to ensure catching
config_data['training']['epochs'] = 0

config = TorchConfig(**config_data)

