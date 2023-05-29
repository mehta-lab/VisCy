# Inference

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

## Config

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

## Config Example

Some working config examples can be found at:

```buildoutcfg
/hpc/projects/CompMicro/projects/virtualstaining/torch_microDL/config_files/2022_HEK_nuc_mem_Soorya/TestData_HEK_2022_04_16/
```

## Single Sample Prediction
It is sometimes the case that inference needs to be run on individual samples. This can be performed by using the `predict_image` method of the `TorchPredictor` object.

Initializing the `TorchPredictor` object for this task requires a config dictionary specifying the model to use for prediction:
> **model_dir:** `absolute path` (Path to parent directory of _pre-trained_ model to use for inference)
>
> **model_name:** `str` (name of model state dict "*.ckpt")

A simple example:

```python
import numpy as np
import sys
sys.path.insert(0, "/pathtoyourinstallation/microDL") #change to your directory
import micro_dl.inference.inference as inference

config = {
    "model_dir": "/hpc/projects/CompMicro/projects/virtualstaining/torch_microDL/models/2023_04_05_Phase2Nuc_HEK_lightning/shalin/lightning_logs/20230408-145505/", #example training dir
    "model_name": "epoch=62-step=6048.ckpt", #example checkpoint
}

# Initialize and run a predictor
torch_predictor = inference.TorchPredictor(
    config=config,
    device="cpu", #'cpu', 'cuda', 'cuda:(int)'
    single_prediction=True,
)
torch_predictor.load_model()

# load your sample and run predictor (random init for example)
sample_input = np.random.rand(1,1,5,512,512)
sample_prediction = torch_predictor.predict_image(sample_input)

```

## Exporting models to onnx

If you wish to run inference via usage of the ONNXruntime, models can be exported to onnx using the `micro_dl/cli/onnx_export_script.py`. See below for an example usage of this script with 5-input-stack model:

```bash
python micro_dl/cli/onnx_export_script.py --model_path path/to/your/pt_model.pt --stack_depth 5 --export_path intended/path/to/model/export.onnx --test_input path/to/test/input.npy
```

**Some Notes:**

* For cpu sharing reasons, running an onnx model requires a dedicated node on hpc OR a non-distributed system (for example a personal laptop or other device).
* Test inputs are optional, but help verify that the exported model can be run if exporting from intended usage device.
* Models must be located in a lighting training logs directory with a valid `config.yaml` in order to be initialized. This can be "hacked" by locating the config in a directory called `checkpoints` beneath a valid config's directory.