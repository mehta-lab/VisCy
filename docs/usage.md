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

Or to preprocess all channels with the default sampling rate and 1 worker:

```sh
viscy preprocess --data_path /path/to/data.zarr
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
