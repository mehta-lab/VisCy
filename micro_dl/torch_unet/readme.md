Training and inference can be performed by using the code in this module, or through the command line (see [scripts](../cli/))<br>
<br>
You can get started by pulling the repository and running these commands in the microDL directory:<br>
```export PYTHONPATH="${PYTHONPATH}:$(pwd)"``` <br>
```python /home/christian.foley/virtual_staining/microDL/micro_dl/cli/torch_train_script.py --config /hpc/projects/CompMicro/projects/virtualstaining/torch_microDL/config_files/2019_02_15_KidneyTissue_DLMBL_subset/torch_config.yml```<br>
```python /home/christian.foley/virtual_staining/microDL/micro_dl/cli/torch_inference_script.py --config /hpc/projects/CompMicro/projects/virtualstaining/torch_microDL/config_files/2019_02_15_KidneyTissue_DLMBL_subset/torch_config.yml```<br>
The code works by making the PyTorch usage as modular as possible, and as a result the ```torch_config.yml```<br>
read by the torch [training](../cli/torch_train_script.py) and [inference](../cli/torch_inference_script.py) scripts <br>
necessarily must access valid preprocessing, training, and inference scripts that would work if used in succession in <br>
the tensorflow version of microDL. <br>
<br>
The structure of the ```torch_config.yml``` config file is as follows: <br>
<br>
preprocess_config_path: absolute path to preprocessing config file used in data preprocessing <br>
train_config_path: absolute path to train config file containing all parameters given in example <br>
inference_config_path: absolute path to inference config file containing all parameters given in example <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* Note that inference runs exactly like tensorflow inference, so all config params are the same except <br> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; it **must** use **data_split: all**, and a **save_folder_name** directory where predictions will be saved <br>
model: <br>
  architecture: 2.5D or 2D (2D currently unstable. Pleasre use 2.5D)<br>
  conv_mode: same (must be same. to be removed in next push) <br>
  in_channels: 1 (number of channels in. If only using phase images, this is 1)<br>
  out_channels: 1 (If only predicting fluorescence, this is 1)<br>
  out_size: (deprecated parameter, to be removed in next push) <br>
  - 1 <br>
  - 2048 <br>
  - 2048 <br>
  residual: true (whether network is residual)<br>
  task: reg (regression or segmentation)<br>
  model_dir: absolute path to *saved pre-trained model*. This is used in inference. You can leave this field empty for training.<br>
training: <br>
  epochs: 41 (number of epochs to train)<br>
  learning_rate: 0.001 (optimizer learning rate)<br>
  optimizer: adam or sgd (optimizer choice)<br>
  loss: mse, l1, cossim (cosine similarity), cel (cross entropy) (loss type to use for training and testing)<br>
  testing_stride: 1 (stride by which to test. Runs validation on testing set every 'testing_stride' epochs) <br>
  save_model_stride: 10 (stride by which to save model. Saves model weights every 'tsave_model_stride' epochs)<br> 
  save_dir: absolute path to directory in which training models and metrics will be saved<br>
  mask: True or False (Whether or not to use masking in training. This is almost always True)<br>
  mask_type: 'rosin'/'unimodal' or 'otsu' (Masking type if above param is True)<br>
  device: 'gpu' or 'cpu'(Device to run training and inference on. Almost always 'gpu')<br>