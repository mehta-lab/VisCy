# Testing workflow with the PyTorch Implementation

Training and inference can be performed by using the code in this module, or through the command line (see [scripts](../cli/)). The dependencies required to run these ```torch_*.py``` scripts can be found in the group conda environment ```microdl_torch.yml```, as well as listed in the conda environment ```torch_conda_env.yml``` in the microDL home directory. Generating a new environment from this file may take a while...

Training models and predictions will be saved in the specified model folder. Currently, to assist testing, intermediate models are saved at a specified frequency (see), and one test prediction is saved from every epoch.

<br>

## Getting Started

Preprocessing is done normally -- It does not depend on pytorch or tensorflow-gpu.
You can get started with training and inference by pulling the repository and running these commands in the microDL directory:

* ```export PYTHONPATH="${PYTHONPATH}:$(pwd)"```
* ```python /home/christian.foley/virtual_staining/microDL/micro_dl/cli/preprocess_script.py --config /hpc/projects/CompMicro/projects/virtualstaining/torch_microDL/config_files/2019_02_15_KidneyTissue_DLMBL_subset/torch_config.yml```
* ```python /home/christian.foley/virtual_staining/microDL/micro_dl/cli/torch_train_script.py --config /hpc/projects/CompMicro/projects/virtualstaining/torch_microDL/config_files/2019_02_15_KidneyTissue_DLMBL_subset/torch_config.yml```
* ```python /home/christian.foley/virtual_staining/microDL/micro_dl/cli/torch_inference_script.py --config /hpc/projects/CompMicro/projects/virtualstaining/torch_microDL/config_files/2019_02_15_KidneyTissue_DLMBL_subset/torch_config.yml```

You will have to replace 'christian.foley' with the user directory you are running microDL from.
<br><br>

## Structure of ```torch_config.yml```

The ```torch_config.yml``` config file contains the parameters for model initiation and training. This config file should be used (as exemplified above) for both training and inference.

<br>

>(**mandatory**, <span style="color:yellow">optional</span>): description

>**zarr_dir**: absolute path to HCS-compatible zarr store containing data
>

>**dataset:**
>
>&nbsp;&nbsp; **input_channels:** <span style="color:cyan"> [1]</span> (list of input channel indices)
>
>&nbsp;&nbsp; **target_channels:** <span style="color:cyan"> [0,3]</span> (list of target channel indices)
>
>&nbsp;&nbsp; **window_size:** <span style="color:cyan"> (5,256,256)</span> (tuple tile size to retrieve from data in form (z,y,x). If network is 2D, z=1)
>
>&nbsp;&nbsp; **normalization:** 
>
>&nbsp;&nbsp;&nbsp;&nbsp; **scheme:** <span style="color:cyan"> "FOV"</span> (normalization scheme, aka what data to include, one of "dataset", "FOV", "tile") 
>
>&nbsp;&nbsp;&nbsp;&nbsp; **type:** <span style="color:cyan"> "median_and_iqr"</span> (Type of normalization to compute. One of "median_and_iqr", "mean_and_std")
>
>&nbsp;&nbsp; **mask_type:** <span style="color:cyan"> 'unimodal'</span> (type of masking to use. One of 'unimodal', 'otsu')
>
>&nbsp;&nbsp; **flatfield_correct:** <span style="color:cyan"> False</span> (whether or not to apply flatfield correction to samples in training)
>
>&nbsp;&nbsp; **batch_size:** <span style="color:cyan"> 8</span> (batch size for sampling dataset)
>
>&nbsp;&nbsp; **split_ratio:** (Splits must must add up to 1. Overriden by use_recorded_data_split = True)
>
>&nbsp;&nbsp;&nbsp;&nbsp; **train:** <span style="color:cyan"> 0.8</span> (portion of data to use for training) 
>
>&nbsp;&nbsp;&nbsp;&nbsp; **val:** <span style="color:cyan"> 0.1</span> (portion of data to use for validation) 
>
>&nbsp;&nbsp;&nbsp;&nbsp; **test:** <span style="color:cyan"> 0.1</span> (portion of data to reserve for testing)
>
>&nbsp;&nbsp;&nbsp;&nbsp; **use_recorded_data_split:** <span style="color:cyan"> False</span> (Whether or not to search for a recorded data split, or randomly split positions)
>

>**model:**
>
>&nbsp;&nbsp; **architecture:** <span style="color:cyan"> 2.5D or 2D</span> 
>
>&nbsp;&nbsp; **in_channels:** <span style="color:cyan"> 1 </span> (number of channels in. If only using phase images, this is 1)
>
>&nbsp;&nbsp; **out_channels:** <span style="color:cyan"> 1 </span> (If only predicting fluorescence, this is 1)
>
>&nbsp;&nbsp; **residual:** <span style="color:cyan"> true </span> (whether network is residual)
>
>&nbsp;&nbsp; **task:** <span style="color:cyan"> reg </span> (regression or segmentation)
>
>&nbsp;&nbsp; **model_dir:** <span style="color:cyan"> absolute path </span> (Path to *saved checkpoint model*; this field is to allow for continuation of training from a checkpoint)
>
>&nbsp;&nbsp; <span style="color:yellow">debug_mode:</span> <span style="color:cyan"> false </span> (If true, running inference will log (save feature maps for) the inference datapath of one input)
>

>**preprocessing:**
>
>&nbsp;&nbsp; **flatfield:** 
>
>&nbsp;&nbsp;&nbsp;&nbsp; **channel_ids** <span style="color:cyan"> [3]</span> (indices of channels to run flatfield correction on, if -1 will use all)
>
>&nbsp;&nbsp;&nbsp;&nbsp; **slice_ids** <span style="color:cyan"> -1 </span> (indices of slices to run flatfield correction on, if -1 will use all)
>
>&nbsp;&nbsp;&nbsp;&nbsp; **block_size** <span style="color:cyan"> 32</span> (chunk size for undersampling in pixels; lower -> longer but more accurate)
>
>&nbsp;&nbsp; **normalize:** (if included will calculate all normalization statistics for specified channels/slices)
>
>&nbsp;&nbsp;&nbsp;&nbsp; **channel_ids** <span style="color:cyan"> [3]</span> (indices of channels to run flatfield correction on, if -1 will use all)
>
>&nbsp;&nbsp;&nbsp;&nbsp; **block_size** <span style="color:cyan"> 32</span> (chunk size for undersampling in pixels; lower -> longer but more accurate)
>
>&nbsp;&nbsp;&nbsp;&nbsp; **num_workers** <span style="color:cyan"> 8</span> (number of workers for multiprocessing)
>
>&nbsp;&nbsp; **masks:** 
>
>&nbsp;&nbsp;&nbsp;&nbsp; **channel_ids** <span style="color:cyan"> [3]</span> (indices of channels to generate masks for, if -1 will use all)
>
>&nbsp;&nbsp;&nbsp;&nbsp; **time_ids** <span style="color:cyan"> -1 </span> (indices of timepoints to generate masks for, if -1 will use all)
>
>&nbsp;&nbsp;&nbsp;&nbsp; **slice_ids** <span style="color:cyan"> -1 </span> (indices of slices to generate masks for, if -1 will use all)
>
>&nbsp;&nbsp;&nbsp;&nbsp; **num_workers** <span style="color:cyan"> 8</span> (number of workers for multiprocessing)
>
>&nbsp;&nbsp;&nbsp;&nbsp; **structuring_element_radius** <span style="color:cyan"> 5</span> (structuring element for mask generation)
>
>&nbsp;&nbsp;&nbsp;&nbsp; **thresholding_type** <span style="color:cyan"> "unimodal"</span> (type of thresholding for masks)
>
>&nbsp;&nbsp;&nbsp;&nbsp; **output_channel** <span style="color:cyan"> None</span> (int index of specific output channel in zarr store to write to. Use this to overwrite past masks if you're computing new ones. Otherwise, you can leave this None. Be wary to not overwrite data.)

>
>**training:**
>
>&nbsp;&nbsp; **epochs:** <span style="color:cyan"> 41 </span> (number of epochs to train)
>
>&nbsp;&nbsp; **samples_per_epoch:** <span style="color:cyan"> 100 </span> (number of samples to use in each epoch. If set to 0 or None, will be automatically computed based off of dataset and window size)
>
>&nbsp;&nbsp; **learning_rate:** <span style="color:cyan"> 0.001 </span> (optimizer learning rate)
>
>&nbsp;&nbsp; **optimizer:** <span style="color:cyan"> adam or sgd </span> (optimizer choice)
>
>&nbsp;&nbsp; **loss:** <span style="color:cyan"> mse, l1, cossim (cosine similarity), cel (cross entropy) </span> (loss type to use for training and testing)
>
>&nbsp;&nbsp; **testing_stride:** <span style="color:cyan"> 41 </span> (stride by which to test. Runs testing on testing set every 'testing_stride' epochs. Feel free to set this to num_epochs, as validation is run every epoch)
>
>&nbsp;&nbsp; **save_model_stride:** <span style="color:cyan"> 10 </span> (stride by which to save model checkpoints; aka, saves a snapshot of model weights every 'save_model_stride' epochs)
>
>&nbsp;&nbsp; **save_dir:** <span style="color:cyan"> absolute path </span> (path to directory in which training models, metadata, and metrics will be saved)
>
>&nbsp;&nbsp; <span style="color:yellow"> **mask:** </span> <span style="color:cyan"> True or False </span> (Whether or not to use masking in training. This is almost always False, and only applies to segmentation models)
>
>&nbsp;&nbsp; <span style="color:yellow"> **mask_type:** </span> <span style="color:cyan"> 'rosin'/'unimodal' or 'otsu' </span> (Masking type if above param is True)
>
>&nbsp;&nbsp; **device:** <span style="color:cyan"> 'gpu' or 'cpu' or int</span> (Device to run training and inference on. Almost always 'gpu' or 0)
>
>&nbsp;&nbsp; <span style="color:yellow"> **num_workers:** </span> <span style="color:cyan"> int </span> (Number of CPU threads used for dataloading. By default will not parallelize dataloading, but it is highly recommended, especially for large samples_per_epoch.)
>
> &nbsp;&nbsp; **augmentations** TODO (for now see example and [docstrings](../input/gunpowder_nodes.py))


## Example Config files

An example config file can be found here:
```/hpc/projects/CompMicro/projects/virtualstaining/torch_microDL/config_files/2022_11_01_VeroMemNuclStain/gunpowder_testing_12_13/torch_config_25D_example.yml```
