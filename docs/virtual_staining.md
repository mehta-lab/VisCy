# Virtual staining of cellular compartments from label-free images

Predicting sub-cellular landmarks such as nuclei and membrane from label-free (e.g. phase) images
can improve imaging throughput and ease experiment design.
However, training a model directly for segmentation requires laborious manual annotation.
We use fluorescent markers as a proxy of supervision with human-annotated labels,
and turn this instance segmentation problem into a paired image-to-image translation (I2I) problem.

VisCy features an end-to-end pipeline to design, train and evaluate I2I models in a declarative manner.
It supports 2D, 2.5D (3D encoder, 2D decoder) and 3D U-Nets,
as well as 3D networks with anisotropic filters (UNeXt2).

## Overview of the pipeline

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

## Model architectures

Reported in the [2024 preprint](https://www.biorxiv.org/content/10.1101/2024.05.31.596901):

Reported in the [2020 paper](https://elifesciences.org/articles/55502v1):

![2.5D U-Net light](https://github.com/mehta-lab/VisCy/blob/main/docs/figures/2_5d_unet_dark.svg?raw=true#gh-light-mode-only)
![2.5D U-Net dark](https://github.com/mehta-lab/VisCy/blob/main/docs/figures/2_5d_unet_dark.svg?raw=true#gh-dark-mode-only)
