# VisCy

VisCy is a deep learning pipeline for training and deploying computer vision models for image-based phenotyping at single cell resolution.

The current focus of the pipeline is on the image translation models for virtual staining of multiple cellular compartments from label-free images.
We are building these models for simultaneous segmentation of nuclei and membrane, which are the first steps in a single-cell phenotyping pipeline. 

## VSCyto2D

## VSCyto3D

## VSNeuromast

## Reference

This pipeline reports the models and training protocols reported in our recent [preprint on robust virtual staining](https://www.biorxiv.org/content/10.1101/2024.05.31.596901):

```bibtex
@article {Liu2024.05.31.596901,
    author = {Liu, Ziwen and Hirata-Miyasaki, Eduardo and Pradeep, Soorya and Rahm, Johanna and Foley, Christian and Chandler, Talon and Ivanov, Ivan and Woosley, Hunter and Lao, Tiger and Balasubramanian, Akilandeswari and Liu, Chad and Leonetti, Manu and Arias, Carolina and Jacobo, Adrian and Mehta, Shalin B.},
    title = {Robust virtual staining of landmark organelles},
    elocation-id = {2024.05.31.596901},
    year = {2024},
    doi = {10.1101/2024.05.31.596901},
    publisher = {Cold Spring Harbor Laboratory},
    URL = {https://www.biorxiv.org/content/early/2024/06/03/2024.05.31.596901},
    eprint = {https://www.biorxiv.org/content/early/2024/06/03/2024.05.31.596901.full.pdf},
    journal = {bioRxiv}
}
```

This pipeline evolved from the [TensorFlow version of virtual staining pipeline](https://github.com/mehta-lab/microDL), which we reported in [this paper in 2020](https://elifesciences.org/articles/55502):

```bibtex
@article {10.7554/eLife.55502,
article_type = {journal},
title = {Revealing architectural order with quantitative label-free imaging and deep learning},
author = {Guo, Syuan-Ming and Yeh, Li-Hao and Folkesson, Jenny and Ivanov, Ivan E and Krishnan, Anitha P and Keefe, Matthew G and Hashemi, Ezzat and Shin, David and Chhun, Bryant B and Cho, Nathan H and Leonetti, Manuel D and Han, May H and Nowakowski, Tomasz J and Mehta, Shalin B},
editor = {Forstmann, Birte and Malhotra, Vivek and Van Valen, David},
volume = 9,
year = 2020,
month = {jul},
pub_date = {2020-07-27},
pages = {e55502},
citation = {eLife 2020;9:e55502},
doi = {10.7554/eLife.55502},
url = {https://doi.org/10.7554/eLife.55502},
keywords = {label-free imaging, inverse algorithms, deep learning, human tissue, polarization, phase},
journal = {eLife},
issn = {2050-084X},
publisher = {eLife Sciences Publications, Ltd},
}
```

 The previous pipeline is now a public archive, and we will be focusing our efforts on VisCy. Our pipeline also provides utilities to export the models to ONNX format for use at runtime.


The following methods are currently in development:
- Virtual staining and supervised learning of of cell state, such as state of infection.
- Self-supervised learning of the cell and organelle states.

Expect rough edges when using above models.


## Installing viscy

1. We highly encourage using a new Conda/virtual environment.

    ```sh
    conda create --name viscy python=3.10
    # OR specify a custom path since the dependencies are large:
    # conda create --prefix /path/to/conda/envs/viscy python=3.10
    ```

2. Clone this repository and install with pip:

    ```sh
    git clone https://github.com/mehta-lab/VisCy.git
    # change to project root directory (parent folder of pyproject.toml)
    cd VisCy
    pip install .
    ```

    If evaluating virtually stained images for segmentation tasks,
    install additional dependencies:

    ```sh
    pip install ".[metrics]"
    ```

    Visualizing the model architecture requires `visual` dependencies:

    ```sh
    pip install ".[visual]"
    ```

3. Verify installation by accessing the CLI help message:

    ```sh
    viscy --help
    ```

For development installation, see [the contributing guide](CONTRIBUTING.md).

The pipeline is built using the [PyTorch Lightning](https://www.pytorchlightning.ai/index.html) framework.
The [iohub](https://github.com/czbiohub-sf/iohub) library is used
for reading and writing data in [OME-Zarr](https://www.nature.com/articles/s41592-021-01326-w) format.

The full functionality is only tested on Linux `x86_64` with NVIDIA Ampere GPUs (CUDA 12.4).
Some features (e.g. mixed precision and distributed training) may not work with other setups,
see [PyTorch documentation](https://pytorch.org) for details.

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
        Cellpose ~~~ aicssegmentation
    end
    input[(Raw Images)] --> sp --> stage{"Training?"}
    stage -.- no -.-> model{{Virtual Staining Model}}
    stage -- yes --> viscy
    viscy --> model
    model --> vs[(Predicted Images)]
    vs --> Segmentation --> output[Biological Analysis]
```

### Model architectures

Reported in the [2024 preprint](https://www.biorxiv.org/content/10.1101/2024.05.31.596901):

Reported in the [2020 paper](https://elifesciences.org/articles/55502v1):
![2.5D U-Net light](https://github.com/mehta-lab/VisCy/blob/main/docs/figures/2_5d_unet_dark.svg?raw=true#gh-light-mode-only)
![2.5D U-Net dark](https://github.com/mehta-lab/VisCy/blob/main/docs/figures/2_5d_unet_dark.svg?raw=true#gh-dark-mode-only)