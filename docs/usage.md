# Using the VisCy CLI

Preprocessing, training, testing, inference, and deployment
can be performed with the `viscy` CLI.

See `viscy --help` for a list of available commands and their help messages.

## Preprocessing

Compute intensity statistics of a dataset
(mean, standard deviation, median, inter-quartile range)
and save them to Zarr metadata.

```sh
viscy preprocess -c config.yaml
```

An example of the config file can be found [here](../examples/configs/preprocess_example.yml).

The are only a few arguments for this command,
so it may be desirable to run without having to edit a config file.
To preprocess all channels with the default sampling rate and 8 workers:

```sh
viscy preprocess --data_path=/path/to/data.zarr --num_workers=8
```

## Training

Training a model is done with the main CLI:

```sh
viscy fit -c config.yaml
```

An example of the config file can be found [here](../examples/configs/fit_example.yml).

By default, TensorBoard logs and checkpoints are saved
in the `default_root_dir/lightning_logs/` directory.

## Testing

This tests a model with regression metrics by default.
For segmentation metrics,
supply ground truth masks and a CellPose segmentation model.

```sh
viscy test -c config.yaml
```

An example of the config file can be found [here](../examples/configs/test_example.yml).

## Inference

Run inference on a dataset and save the results to OME-Zarr:

```sh
viscy predict -c config.yaml
```

An example of the config file can be found [here](../examples/configs/predict_example.yml).

## Deployment

Export model to ONNX format for deployment:

```sh
viscy export -c config.yaml
```

An example of the config file can be found [here](../examples/configs/export_example.yml).

Use argument `export_path` to configure where the output is stored.

### Notes

* Current implementation will export a checkpoint to ONNX IR version 9
and OP set version 18.

* For CPU sharing reasons, running an ONNX model
requires an exclusive node on HPC OR a non-distributed system (e.g. a PC).

* Models must be located in a lightning training logs directory
with a valid `config.yaml` in order to be initialized.
This can be "hacked" by locating the config in a directory
called `checkpoints` beneath a valid config's directory.
