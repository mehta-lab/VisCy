## Inference

The main command for inference is:
```buildoutcfg
python micro_dl/cli/torchinference_script.py --config <config path (.yml)> --gpu <gpu id (default 0)> --gpu_mem_frac <0-1 (default 1>
```

where the parameters are defined as follows:
* **config** (yaml file): Configuration file, see below.
* **gpu** (int): ID number of if you'd like to specify which GPU you'd like to run on. If you don't
specify a GPU then the GPU with the largest amount of available memory will be selected for you.
* **gpu_mem_fraction** (float): You can specify what fraction of total GPU memory you'd like to utilize.
If there's not enough memory available on the GPU, and AssertionError will be raised.
If memory fraction is unspecified, all memory currently available on the GPU will automatically
be allocated for you.

# Config

> **zarr_dir**: `absolute path` (absolute path to HCS-compatible zarr store containing data)
>
> **model_dir:** `absolute path` (Path to parent directory of _pre-trained_ model to use for inference)
>
> **model_name:** `str` (name of model state dict "*.ckpt")
>
> **batch_size:** `int` (size of batch for batch prediction: needs to be small enough to fit on GPU, but larger is faster)
>
> **time_indices:** `list[int]` (list of time indices to predict)
>
> **device:** `str` ("cpu", "cuda", or "i" i=int of gpu to use)
>
> **positions:** `nested dict` (nested dict of positions to run inference on inside your zarr store, see examples for format)
>
> **normalize_inputs:** `{true, false}` (whether to normalize inputs; (should reflect whether this was done when training the model))
>
> ***norm_type:*** `{"median_and_iqr","mean_and_std"}` (normalization type used for training (Only used if normalize_inputs is true))
>
> ***norm_scheme:*** `{"dataset","FOV"}` (normalization scheme used for training (Only used if normalize_inputs is true))
>
> **input_channels:** `list[str]` (list of input channels by name)
>
> **save_preds_to_model_dir:** `True` (Whether or not to save predictions to model directory)
>
> ***custom_save_preds_dir:*** `absolute path` (Path to custom save directory. Generally try to avoid using this, since it delocates model predictions from the models)

# Config Example
Some working config examples can be found at:
```buildoutcfg
/hpc/projects/CompMicro/projects/virtualstaining/torch_microDL/config_files/2022_HEK_nuc_mem_Soorya/TestData_HEK_2022_04_16/
```

#TODO: evaluation script and config