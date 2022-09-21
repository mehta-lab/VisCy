# %%
'''
The preprocessing config :
/gpfs/CompMicro/projects/virtualstaining/tf_microDL/config/2022_08_01_A549RickettsiaCytotoxicityTest/config_preprocess_A549Cytotoxicity_set2.yml
The training config:
/gpfs/CompMicro/projects/virtualstaining/tf_microDL/config/2022_08_01_A549RickettsiaCytotoxicityTest/config_train_A549Cytotoxicity_train5.yml
The inference config file:
/gpfs/CompMicro/projects/virtualstaining/tf_microDL/config/2022_08_01_A549RickettsiaCytotoxicityTest/config_inference_A549NuclMem_testHEKset2.yml
Dataset A :
Trained model:
/CompMicro/projects/Rickettsia/2022_RickettsiaAnalysis_Soorya/3_Cell_Image_Preprocessing/VirtualStainingMicroDL_A549_2022_08_17/Translation_temp_6/
Dataset B1:
/gpfs/CompMicro/projects/Rickettsia/2022_RickettsiaAnalysis_Soorya/3_Cell_Image_Preprocessing/VirtualStainingMicroDL_A549NuclMem_2022_09_14_15/TestInferenceData_HEK1/
Dataset B2:
/gpfs/CompMicro/projects/Rickettsia/2022_RickettsiaAnalysis_Soorya/3_Cell_Image_Preprocessing/VirtualStainingMicroDL_A549NuclMem_2022_09_14_15/TestInferenceData_HEK2/
'''
# %%
import sys
import os

# Add module path to sys
module_path = os.path.abspath(os.path.join('..'))
print("System path: "+module_path)
if module_path not in sys.path:
    sys.path.append(module_path)

# %%
import glob

import yaml

from micro_dl.inference import image_inference as image_inf
import micro_dl.utils.train_utils as train_utils
import micro_dl.utils.preprocess_utils as preprocess_utils

# %%
with open('/hpc/projects/CompMicro/projects/virtualstaining/tf_microDL/config/2022_08_01_A549RickettsiaCytotoxicityTest/config_train_A549Cytotoxicity_train5.yml', 'r') as f:
    train_config = yaml.safe_load(f)
with open('/hpc/projects/CompMicro/projects/virtualstaining/tf_microDL/config/2022_08_01_A549RickettsiaCytotoxicityTest/config_inference_A549NuclMem_testHEKset2.yml', 'r') as f:
    inference_config = yaml.safe_load(f)
with open('/hpc/projects/CompMicro/projects/virtualstaining/tf_microDL/config/2022_08_01_A549RickettsiaCytotoxicityTest/config_preprocess_A549Cytotoxicity_set2.yml', 'r') as f:
    preprocess_config = yaml.safe_load(f)
# %%
print(preprocess_config)
train_config['dataset']['data_dir'] = train_config['dataset']['data_dir'].replace('/gpfs', '/hpc/projects')
train_config['trainer']['model_dir'] = train_config['trainer']['model_dir'].replace('/gpfs', '/hpc/projects')
inference_config['preprocess_dir'] = inference_config['preprocess_dir'].replace('/gpfs', '/hpc/projects')
inference_config['image_dir'] = inference_config['image_dir'].replace('/gpfs', '/hpc/projects')
inference_config['model_dir'] = inference_config['model_dir'].replace('/gpfs', '/hpc/projects')
# %%
print('height' in [key for key in train_config['network']])
print('width' in [key for key in train_config['network']])
#%%
gpu_ids = 0
gpu_mem_frac = 0.95
inference_inst = image_inf.ImagePredictor(
            train_config=train_config,
            inference_config=inference_config,
            preprocess_config=preprocess_config,
            gpu_id=gpu_ids,
            gpu_mem_frac=gpu_mem_frac,
        )
inference_inst.run_prediction()