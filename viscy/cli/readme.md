# CLI

## Exporting models to ONNX

If you wish to run inference via usage of the ONNXruntime, models can be exported to onnx using the `micro_dl/cli/onnx_export_script.py`. See below for an example usage of this script with 5-input-stack model:

```bash
python micro_dl/cli/onnx_export_script.py --model_path path/to/your/pt_model.pt --stack_depth 5 --export_path intended/path/to/model/export.onnx --test_input path/to/test/input.npy
```

**Some Notes:**

* For cpu sharing reasons, running an onnx model requires a dedicated node on hpc OR a non-distributed system (for example a personal laptop or other device).
* Test inputs are optional, but help verify that the exported model can be run if exporting from intended usage device.
* Models must be located in a lighting training logs directory with a valid `config.yaml` in order to be initialized. This can be "hacked" by locating the config in a directory called `checkpoints` beneath a valid config's directory.