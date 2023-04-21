# Gunpowder Dataloading Documentation

The gunpowder dataloading branch allows for the usage of a single config file containing all our parameters to run preprocessing, training, and inference on the same OME-NGFF HCS-compatible zarr store in sequence. The backed mainly uses [Gunpowder](https://github.com/funkey/gunpowder) and [PyTorch](https://pytorch.org/) to perform randomized dataloading, augmentation, training, and inference using deep convolutional neural networks to spec.

This pipeline can be run through the command line (see [scripts](../cli/)). The dependencies required to run these scripts (```torch_*.py``` and ```preprocessing_script.py```) can be found in the comp_micro group conda environment ```microdl_torch.yml``` on ESS, as well as listed in the conda environment ```torch_conda_env.yml``` in the microDL home directory. Generating a new environment from this file may take a while...

Training models and predictions will be saved in the specified model folder. Currently, to assist testing, intermediate models are saved at a specified frequency (see), and one test prediction is saved from every epoch.

<br>

## Getting Started

Preprocessing is done normally -- It does not depend on pytorch or tensorflow-gpu.
You can get started with training and inference by pulling the repository and running these commands in the microDL directory:

* ```export PYTHONPATH="${PYTHONPATH}:$(pwd)"```
* ```python /home/christian.foley/virtual_staining/microDL/micro_dl/cli/preprocessing_script.py --config your_config_file_path.yml```
* ```python /home/christian.foley/virtual_staining/microDL/micro_dl/cli/torch_train_script.py --config your_config_file_path.yml```
* ```python /home/christian.foley/virtual_staining/microDL/micro_dl/cli/torch_inference_script.py --config your_config_file_path.yml```

You will have to replace 'christian.foley' with the user directory you are running microDL from.

<br><br>

## Structure of ```your_config_file_path.yml```

The ```your_config_file_path.yml``` config file contains the necessary parameters for every step of the pipeline. This config file should be used (as exemplified above) for preprocessing, training, and inference.

<br>

>(**mandatory**, <span style="color:yellow">optional</span>): description

>**zarr_dir:** absolute path to the top directory of the OME-NGFF HCS-compatible zarr store containing your data. This store will be modified in place, so it should *not* be the only copy.
>
> **dataset:**
>
>&nbsp;&nbsp; **input_channels:** <span style="color:cyan"> [1]</span> (list of input channel indices to the model) 
>
>&nbsp;&nbsp; **target_channels:** <span style="color:cyan"> [0]</span> (list of target channel indices to the model) 
>
>&nbsp;&nbsp; **window_size:** <span style="color:cyan"> (5,256,256)</span> (size of the spatial windows which gunpowder samples from data. This should be the max size across each dimension, (z,y,x). If you have a 2D model, simply request (1,256,256). If you have a 3d model which needs 7 slices of input data, request (7, 256, 256), etc.) 
>
>&nbsp;&nbsp; **normalization:** 
>
>&nbsp;&nbsp;&nbsp;&nbsp; **scheme:** <span style="color:cyan"> 'FOV'</span> (Normalization scheme to apply during preprocessing, training and inference. One of 'dataset', 'FOV', 'tile'. Data will be normalized to the values calculated across this range of the dataset)
>
>&nbsp;&nbsp;&nbsp;&nbsp; **type:** <span style="color:cyan"> 'median_and_iqr'</span> (type of normalization to apply during preprocessing, training and inference. One of 'median_and_iqr', 'mean_and_std'. We have found that median and iqr give a better representation of the dynamic range of biological data)
>
>&nbsp;&nbsp; **min_foreground_fraction:** <span style="color:cyan"> 0.1 </span> (Minimum fraction of foreground information in a sample to be allowed to pass through pipeline. If set to 0, all samples will be allowed)
>
>&nbsp;&nbsp; **mask_type:** <span style="color:cyan"> 'otsu' </span> (algorithm for generation mask thresholds. one of 'otsu', 'unimodal')
>
>&nbsp;&nbsp; **flatfield_correct:** <span style="color:cyan"> True </span> (Whether to apply flatfield correction to samples in the pipeline. Only set to True if you have calculated flatfields in preprocessing)
>
>&nbsp;&nbsp; **batch_size:** <span style="color:cyan"> 64 </span> (Batch size requested from pipeline)
>
>&nbsp;&nbsp; **split_ratio:** 
>
>&nbsp;&nbsp;&nbsp;&nbsp; **test:**<span style="color:cyan"> 0.15 </span> (Fraction of positions used for testing)
>
>&nbsp;&nbsp;&nbsp;&nbsp; **train:**<span style="color:cyan"> 0.7 </span> (Fraction of positions used for training)
>
>&nbsp;&nbsp;&nbsp;&nbsp; **val:**<span style="color:cyan"> 0.15 </span> (Fraction of positions used for validation)
>
>**model:**
>
>&nbsp;&nbsp; **architecture:** <span style="color:cyan"> 2.5D or 2D</span> 
>
>&nbsp;&nbsp; **in_channels:** <span style="color:cyan"> 1 </span> (number of channels in. If only using phase images, this is 1)
>
>&nbsp;&nbsp; **out_channels:** <span style="color:cyan"> 1 </span> (number of channels out. This should match your dataset: target_channels param)
>
>&nbsp;&nbsp; **in_stack_depth:** <span style="color:cyan"> 5 </span> (depth of your stack. This should match the depth specified in the first element of dataset: window_size)
>
>&nbsp;&nbsp; **residual:** <span style="color:cyan"> true </span> (whether network blocks are residual)
>
>&nbsp;&nbsp; **task:** <span style="color:cyan"> reg </span> (regression or segmentation)
>
>&nbsp;&nbsp; **model_dir:** <span style="color:cyan"> absolute path </span> (Path to *saved pre-trained model*; this is used in inference and you can leave this field empty for training and preprocessing)
>
>&nbsp;&nbsp; <span style="color:yellow">debug_mode:</span> <span style="color:cyan"> false </span> (If true, running inference will log (save feature maps for) the inference datapath of one input)
>
>**training:**
>
>&nbsp;&nbsp; **samples_per_epoch:** <span style="color:cyan"> 300 </span> (number of tiles for gunpowder to sample each epoch. If set to 0, will be automatically estimated based off the size of your window and the size of your dataset)
>
>&nbsp;&nbsp; **epochs:** <span style="color:cyan"> 41 </span> (number of epochs to train)
>
>&nbsp;&nbsp; **learning_rate:** <span style="color:cyan"> 0.001 </span> (optimizer learning rate)
>
>&nbsp;&nbsp; **optimizer:** <span style="color:cyan"> adam or sgd </span> (optimizer choice)
>
>&nbsp;&nbsp; **loss:** <span style="color:cyan"> mse, l1, cossim (cosine similarity), cel (cross entropy) </span> (loss type to use for training and testing)
>
>&nbsp;&nbsp; **testing_stride:** <span style="color:cyan"> 1 </span> (stride by which to test. Runs validation on testing set every 'testing_stride' epochs)
>
>&nbsp;&nbsp; **save_model_stride:** <span style="color:cyan"> 1 </span> (stride by which to save model. Saves checkpoint of model weights every 'tsave_model_stride' epochs)
>
>&nbsp;&nbsp; **save_dir:** <span style="color:cyan"> absolute path </span> (path to directory in which training model checkpoints and metrics will be saved)
>
>&nbsp;&nbsp; **device:** <span style="color:cyan"> 'gpu' or 'cpu' or int</span> (Device to run training and inference on. Almost always 'gpu' or 0)
>
>&nbsp;&nbsp; <span style="color:yellow"> num_workers: </span> <span style="color:cyan"> int </span> (Number of CPU threads used for dataloading. By default will not parallelize dataloading. There is some overhead that needs to be done each epoch, so the recommended number is between 2 and 8 depending on the size of dataset.)
>
>&nbsp;&nbsp;  **augmentations:** (the following parameters control the augmentation nodes and hyperparameters in the generated pipeline)
>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **transpose:** 
>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  **transpose_only:** <span style="color:cyan">  [-2,-1] </span> (The dimensions to transpose along. NOTE: these are indexed by the dataset window_size parameter, which dictates only the spatial dimensions of the data. Generally you should leave these alone)
>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **mirror:** 
>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  **mirror_only:** <span style="color:cyan"> [-2,-1] </span> (The dimensions to mirror along. NOTE: these are indexed by the dataset window_size parameter, which dictates only the spatial dimensions of the data. Generally you should leave these alone)
>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **rotate:** 
>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  **rotation_interval:** <span style="color:cyan"> [0,3.14] </span> (The interval from which to randomly sample rotation angles, in radians)
>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **zoom:** 
>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  **scale_interval:** <span style="color:cyan"> [0.7,1.3] </span> (The interval from which to randomly sample spatial scaling coefficients)
>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **blur:** 
>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  **mode:** <span style="color:cyan"> 'gaussian' </span> (type of blur, one of 'gaussian', 'rectangle', 'defocus')
>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  **width_range:** <span style="color:cyan"> [3,7] </span> (The interval from which to randomly sample kernel pixel widths)
>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  **sigma:** <span style="color:cyan"> 0.1 </span> (Sigma for kernel generation if using 'gaussian')
>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  **prob:** <span style="color:cyan"> 0.2 </span> (The probability of blurring any given sample passed through the pipeline)
>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  **blur_channels:** <span style="color:cyan"> [1] </span>(list of channels to blur in any sample chosen for blurring)
>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **shear:** 
>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  **angle_range:** <span style="color:cyan"> [3,7] </span> (The interval from which to randomly sample angles for shearing)
>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  **prob:** <span style="color:cyan"> 0.2 </span> (The probability of shearing any given sample passed through the pipeline)
>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  **shear_middle_slice_only:** <span style="color:cyan"> [1] </span>(list of channels indices to shear the middle slice only. This is used to reduce computation time on channels that only 2D samples are needed from (gunpowder by necessity will still read 3d data but slice it at the end). This, if used, should just be a list of your target channels or any unused channels.)
>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **intensity_jitter:** 
>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  **scale_range:** <span style="color:cyan"> [0.7, 1.3] </span>(The range from which to randomly sample intensity scaling coefficients)
>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  **shift_range:** <span style="color:cyan"> [-0.15, 0.15] </span>(The range from which to randomly sample intensity shifting coefficients)
>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  **norm_before_shift:** <span style="color:cyan"> True </span>(If true, will normalize between -1 and 1 before shifting. This means that your shifting range values are scaled to the intensity of the image (or vice versa))
>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  **jitter_demeaned:** <span style="color:cyan"> True </span>(If true, only applies jitter shift to the demeaned components with the intent to augment _contrast_ over absolute intensity)
>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  **prob:** <span style="color:cyan"> 0.2 </span>(The probability of applying intensity jitter any given sample passed through the pipeline)
>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  **jitter_channels:** <span style="color:cyan"> [1] </span>(list of channels to apply intensity jitter to)
>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **noise:** 
>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  **mode:** <span style="color:cyan"> 'gaussian' </span>(The type of noise to apply. See scikit-image documentation)
>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  **seed:** <span style="color:cyan"> 14156 </span>(Seed for generating random noise)
>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  **clip:** <span style="color:cyan"> True </span>(Whether to clip outliers of noisy sample after application)
>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  **prob:** <span style="color:cyan"> 0.2 </span>(Probabilty of applying noise to any given sample passed through the pipeline)
>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  **noise_channles:** <span style="color:cyan"> [1] </span> (list of channels to apply noise to)

<br>


## Example Config files

Example config files can be found in this directory:
```/hpc/projects/CompMicro/projects/virtualstaining/torch_microDL/config_files/2019_02_15_KidneyTissue_DLMBL_subset```

<br>

## Important parameters to look for

Make sure that these parameters are changed to reflect your session. This will prevent file conflicts and overwriting.

**Preprocessing:**

* **preprocess_config_path** (in torch_config) - Make sure this is pointing to *your* tf preprocessing config

**Training:**

* **train_config_path** (in torch_config) - Make sure this is pointing to *your* tf training config
* **save_dir** (in torch_config/training) - Where the models and training predictions will be saved

**Inference:**

* **inference_config_path** (in torch_config) - Make sure this is pointing to *your* tf inference config
* **save_folder_name** (in config_inference.yml) - Where the inference predictions and metrics will be saved. If not global, dir path will be created with `training_model_yyyy_mm_dd_hh_mm` dir as root.
* **model_dir** (in torch_config/model) - You want this to point to the ```.pt``` model file you just trained
