# DynaCLR

Implementation for ICLR 2025 submission:
Contrastive learning of cell state dynamics in response to perturbations.

## Installation

> **Note**:
> The full functionality is tested on Linux `x86_64` with NVIDIA Ampere/Hopper GPUs (CUDA 12.4).
> The CTC example configs are also tested on macOS with Apple M1 Pro SoCs (macOS 14.7).
> Apple Silicon users need to make sure that they use
> the `arm64` build of Python to use MPS acceleration.
> Tested to work on Linux on the High Performance cluster, and may not work in other environments.
> The commands below assume a Unix-like system.

1. We recommend using a new Conda/virtual environment.

    ```sh
    conda create --name dynaclr python=3.10
    ```

2. Install the package with `pip`:

    ```sh
    conda activate dynaclr
    # in the project root directory
    # i.e. where this README is located
    pip install -e ".[visual,metrics]"
    ```

3. Verify installation by accessing the CLI help message:

    ```sh
    viscy --help
    ```

For development installation, see [the contributing guide](./CONTRIBUTING.md).

## Reproducing DynaCLR

Due to anonymity requirements during the review process,
we cannot host the large custom datasets used in the paper.
Here we demonstrate how to train and evaluate the DynaCLR models with a small public dataset.
Here we use the training split of a HeLa cell DIC dataset from the
[Cell Tracking Challenge](http://data.celltrackingchallenge.net/training-datasets/DIC-C2DH-HeLa.zip)
and convert it to OME-Zarr for convenience (`../Hela_CTC.zarr`).
This dataset has 2 FOVs, and we use a 1:1 split for training and validation.

Verify the dataset download by running the following command.
You may need to modify the path in the configuration file to point to the correct dataset location.

```sh
# modify the path in the configuration file
# to use the correct dataset location
iohub info /path/to/Hela_CTC.zarr
```

It should print something like:

```text
=== Summary ===
Format:    omezarr v0.4
Axes:    T (time); C (channel); Z (space); Y (space); X (space);
Channel names:   ['DIC', 'labels']
Row names:   ['0']
Column names:   ['0']
Wells:    1
```

Training can be performed with the following command:

```sh
python -m viscy.cli.contrastive_triplet fit -c ./examples/fit_ctc.yml
```

The TensorBoard logs and model checkpoints will be saved the `./lightning_logs` directory.

Prediction of features on the entire dataset using the trained model can be done with:

```sh
python -m viscy.cli.contrastive_triplet predict -c ./examples/predict_ctc.yml
```
