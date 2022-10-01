# Testing workflow with the PyTorch Implementation

Training and inference can be performed by using the code in this module, or through the command line (see [scripts](../cli/)). The dependencies required to run these ```torch_*.py``` scripts can be found in the group conda environment ```microdl_torch.yml```, as well as listed in the conda environment ```torch_conda_env.yml``` in the microDL home directory. Generating a new environment from this file may take a while...

Training models and predictions will be saved in the specified model folder. Currently, to assist testing, intermediate models are saved at a specified frequency (see), and one test prediction is saved from every epoch.

<br>

## Getting Started

Preprocessing is done normally -- It does not depend on pytorch or tensorflow-gpu.
You can get started with training and inference by pulling the repository and running these commands in the microDL directory:

* ```export PYTHONPATH="${PYTHONPATH}:$(pwd)"```
* ```python /home/christian.foley/virtual_staining/microDL/micro_dl/cli/torch_train_script.py --config /hpc/projects/CompMicro/projects/virtualstaining/torch_microDL/config_files/2019_02_15_KidneyTissue_DLMBL_subset/torch_config.yml```
* ```python /home/christian.foley/virtual_staining/microDL/micro_dl/cli/torch_inference_script.py --config /hpc/projects/CompMicro/projects/virtualstaining/torch_microDL/config_files/2019_02_15_KidneyTissue_DLMBL_subset/torch_config.yml```

You will have to replace 'christian.foley' with the user directory you are running microDL from.

The code works by making the PyTorch usage as modular as possible, and as a result the ```torch_config.yml``` read by the torch [training](../cli/torch_train_script.py) and [inference](../cli/torch_inference_script.py) scripts necessarily must access valid preprocessing, training, and inference scripts that would work if used in succession in the tensorflow version of microDL.
<br><br>

## Structure of ```torch_config.yml```

The ```torch_config.yml``` config file contains the parameters for model initiation and training. This config file should be used (as exemplified above) for both training and inference.

* *Note: the PyTorch implementation relies heavily on preprocessing and dataloading from tensorflow microDL. Because of this, valid (minus some training parameters) preprocessing, training, and inference configs must be referenced. This will be updated soon*

<br>

**preprocess_config_path:** absolute path to preprocessing config file used in data preprocessing

**train_config_path:** absolute path to train config file containing all parameters given in example

**inference_config_path:** absolute path to inference config file containing all parameters given in example

* Note that inference runs exactly like tensorflow inference, so all config params are the same except it *must* use **data_split: all**, and *should* specify a **save_folder_name** directory where predictions will be saved (this is best practice).

* If no **save_folder_name** directory is specified, the inference script will automatically save parralel to the ```data``` folder specified by ```preprocess_config.yml```

**model:**

&nbsp;&nbsp; **architecture:** <span style="color:cyan"> 2.5D or 2D</span> (2D currently unstable. Pleasre use 2.5D)

&nbsp;&nbsp; **conv_mode:** <span style="color:cyan"> same </span> (must be same. to be removed in next push)

&nbsp;&nbsp; **in_channels:** <span style="color:cyan"> 1 </span> (number of channels in. If only using phase images, this is 1)

&nbsp;&nbsp; **out_channels:** <span style="color:cyan"> 1 </span> (If only predicting fluorescence, this is 1)

&nbsp;&nbsp; **residual:** <span style="color:cyan"> true </span> (whether network is residual)

&nbsp;&nbsp; **task:** <span style="color:cyan"> reg </span> (regression or segmentation)

&nbsp;&nbsp; **model_dir:** <span style="color:cyan"> absolute path </span> (Path to *saved pre-trained model*; this is used in inference and you can leave this field empty for training)

**training:** 

&nbsp;&nbsp; **epochs:** <span style="color:cyan"> 41 </span> (number of epochs to train)

&nbsp;&nbsp; **learning_rate:** <span style="color:cyan"> 0.001 </span> (optimizer learning rate)

&nbsp;&nbsp; **optimizer:** <span style="color:cyan"> adam or sgd </span> (optimizer choice)

&nbsp;&nbsp; **loss:** <span style="color:cyan"> mse, l1, cossim (cosine similarity), cel (cross entropy) </span> (loss type to use for training and testing)

&nbsp;&nbsp; **testing_stride:** <span style="color:cyan"> 1 </span> (stride by which to test. Runs validation on testing set every 'testing_stride' epochs)

&nbsp;&nbsp; **save_model_stride:** <span style="color:cyan"> 10 </span> (stride by which to save model. Saves model weights every 'tsave_model_stride' epochs)

&nbsp;&nbsp; **save_dir:** <span style="color:cyan"> absolute path </span> (path to directory in which training models and metrics will be saved)

&nbsp;&nbsp; **mask:** <span style="color:cyan"> True or False </span> (Whether or not to use masking in training. This is almost always True)

&nbsp;&nbsp; **mask_type:** <span style="color:cyan"> 'rosin'/'unimodal' or 'otsu' </span> (Masking type if above param is True)

&nbsp;&nbsp; **device:** <span style="color:cyan"> 'gpu' or 'cpu'</span> (Device to run training and inference on. Almost always 'gpu')

<br>

## Example Config files

Example config files can be found in this directory:
```/hpc/projects/CompMicro/projects/virtualstaining/torch_microDL/config_files/2019_02_15_KidneyTissue_DLMBL_subset```

## Important items to change

Make sure that these parameters are changed to reflect your session. This will prevent file conflicts and overwriting.

**Preprocessing:**

* **preprocess_config_path** (in torch_config) - Make sure this is pointing to *your* tf preprocessing config

**Training:**

* **train_config_path** (in torch_config) - Make sure this is pointing to *your* tf training config
* **save_dir** (in torch_config/training) - Where the models and training predictions will be saved

**Inference:**

* **inference_config_path** (in torch_config) - Make sure this is pointing to *your* tf inference config
* **save_folder_name** (in config_inference.yml) - Where the inference predictions and metrics will be saved
* **model_dir** (in torch_config/model) - You want this to point to the ```.pt``` model file you just trained
