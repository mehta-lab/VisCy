# DynaCLR

Implementation for NeurIPS 2025 submission:
Contrastive learning of cell state dynamics in response to perturbations.

## Installation

> **Note**:
> The full functionality is tested on Linux `x86_64` with NVIDIA Ampere/Hopper GPUs (CUDA 12.4).
> The CTC example configs are also tested on macOS with Apple M1 Pro SoCs (macOS 14.7).
> Apple Silicon users need to make sure that they use
> the `arm64` build of Python to use MPS acceleration.
> Tested to work on Linux on the High Performance cluster, and may not work in other environments.
> The commands below assume a Unix-like system.

This can be done via the `setup.sh` in the `/examples/DynaCLR/`. Follow instructions in the [examples README.md](/examples/DynaCLR/README.md)


Alternatively, the pacakge can be installed via the following instructions:

1. We recommend using a new Conda/virtual environment.

    ```sh
    conda create --name dynaclr python=3.11
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
