# Using VisCy

This page briefly describes the workflow of training,
evaluating, and deploying a virtual staining model with VisCy.

## Preprocessing

VisCy uses a simple preprocessing script to compute intensity metrics
(mean, standard deviation, median, inter-quartile range)
to normalize the images during training and inference.
Use with:

```sh
python -m vicy.cli.preprocess -c config.yaml
```

An example of the config file is shown below:

```yaml
zarr_dir: /path/to/ome.zarr
preprocessing:
  normalize:
    # index of channels to compute statistics on
    channel_ids: [0, 1, 2]
    # statistics are computed in local blocks
    # avoid high RAM usage
    block_size: 32
    # number of CPU cores to parallelize over
    num_workers: 16
```

> **Note:** This script is subject to change.
> It may be moved into the main CLI in the future.

## CLI

Training, testing, inference, and deployment can be performed with the `viscy` CLI.

See `viscy --help` for a list of available commands and their help messages.

### Training

Training a model is done with the main CLI:

```sh
viscy fit -c config.yaml
```

An example of the config file can be found [here](../examples/configs/fit_example.yml).

By default, TensorBoard logs and checkpoints are saved
in the `default_root_dir/lightning_logs/` directory.

### Testing

This tests a model with regression metrics by default.
For segmentation metrics,
supply ground truth masks and a CellPose segmentation model.

```sh
viscy test -c config.yaml
```

An example of the config file can be found [here](../examples/configs/test_example.yml).

### Inference

Run inference on a dataset and save the results to OME-Zarr:

```sh
viscy predict -c config.yaml
```

An example of the config file can be found [here](../examples/configs/predict_example.yml).

### Deployment

Export model to ONNX format for deployment:

```sh
viscy export -c config.yaml
```

An example of the config file can be found [here](../examples/configs/export_example.yml).
