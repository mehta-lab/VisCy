# viscy

viscy is a deep learning pipeline for training and deploying computer vision models for high-throughput imaging and image-based phenotyping with single cell resolution.

The current focus of the pipeline is on the image translation models for virutal staining of multiple cellular compartments from label-free images. We are building these models for screening fields of view during imaging and for simultaneous segmentation of nuclei and membrane for single-cell phenotyping. The pipeline provides utilities to export the models to onnx format for use during runtime. We will grow the collection of the models suitable for high-throughput imaging and phenotyping.

![virtual_staining](docs/figures/phase_to_nuclei_membrane.svg)

## Installation

(Optional) create a new virtual/Conda environment.

Clone this repository and install viscy:

```sh
git clone https://github.com/mehta-lab/viscy.git
pip install viscy
```

Verify installation by accessing the CLI help message:

```sh
viscy --help
```

For development installation, see [the contributing guide](CONTRIBUTING.md).

The pipeline is built using the [pytorch lightning](https://www.pytorchlightning.ai/index.html) framework and [iohub](https://github.com/czbiohub-sf/iohub) library for reading and writing data in [ome-zarr](https://www.nature.com/articles/s41592-021-01326-w) format.

The full functionality is  tested only on Linux `x86_64` with NVIDIA Ampere GPUs (CUDA 12.0).
Some features (e.g. mixed precision and distributed training) may not work with other setups,
see [PyTorch documentation](https://pytorch.org) for details.

Following dependencies will allow use and development of the pipeline, while the pypi package is pending:

```<yaml>
iohub==0.1.0.dev3
torch>=2.0.0
torchvision>=0.15.1
tensorboard>=2.13.0
lightning>=2.0.1
monai>=1.2.0
jsonargparse[signatures]>=4.20.1
scikit-image>=0.19.2
matplotlib
cellpose==2.1.0
lapsolver==1.1.0
scikit-learn>=1.1.3
scipy>=1.8.0
torchmetrics[detection]>=1.0.0
pytest
pytest-cov
hypothesis
profilehooks
onnxruntime
```

## Virtual staining of cellular compartments from label-free images

Predicting sub-cellular landmarks such as nuclei and membrane from label-free (e.g. phase) images
can improve imaging throughput and ease experiment design.
However, training a model directly for segmentation requires laborious manual annotation.
We use fluorescent markers as a proxy of supervision with human-annotated labels,
and turn this instance segmentation problem into a paired image-to-image translation (I2I) problem.

viscy features an end-to-end pipeline to design, train and evaluate I2I models in a declarative manner.
It supports 2D, 2.5D (3D encoder, 2D decoder) and 3D U-Nets,
as well as 3D networks with anisotropic filters.

### Overview of the pipeline

```mermaid
flowchart LR
    subgraph sp[Signal Processing]
        Registration --> Reconstruction --> Resampling
    end
    subgraph viscy["Computer Vision (viscy)"]
        subgraph Preprocessing
            Normalization -.-> fd[Feature Detection]
        end
        subgraph Training
            arch[Model Architecting]
            hyper[Hyperparameter Tuning]
            val[Performance Validation]
            compute[Acceleration]
            arch <--> hyper <--> compute <--> val <--> arch
        end
        subgraph Testing
            regr[Regression Metrics]
            segm[Instance Segmentation Metrics]
            cp[CellPose]
            cp --> segm
        end
        Preprocessing --> Training --> Testing
        Testing --> test{"Performance?"}
        test -- good --> Deployment
        test -- bad --> Training
    end
    subgraph Segmentation
        CellPose ~~~ aicssegmentation
    end
    input[(Raw Images)] --> sp --> stage{"Training?"}
    stage -.- no -.-> model{{Virtual Staining Model}}
    stage -- yes --> viscy
    viscy --> model
    model --> vs[(Predicted Images)]
    vs --> Segmentation --> output[Biological Analysis]
```

### Model architecture

![2.5D U-Net light](docs/figures/2_5d_unet_light.svg#gh-light-mode-only)
![2.5D U-Net dark](docs/figures/2_5d_unet_dark.svg#gh-dark-mode-only)
