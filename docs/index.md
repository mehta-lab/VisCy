---
hide:
  - navigation
---

# VisCy

A deep learning pipeline for image-based phenotyping at single-cell resolution.[^name]

[^name]:
    **VisCy** is a blend of *vision* and *cyto* (cell) — computer vision for
    cell biology.

!!! quote ""

    :material-microscope:{ .lg .middle } &nbsp; train and deploy
    virtual staining, segmentation, and self-supervised representation models on terabyte-scale imaging datasets.

## What's inside

VisCy is a [uv workspace](https://docs.astral.sh/uv/concepts/workspaces/) monorepo:
consisting of a core of independently versioned **packages** that compose into self-contained
research **applications**.

<div class="grid cards" markdown>

-   :material-database-outline:{ .lg .middle } __viscy-data__

    ---

    Data loading and Lightning `DataModule`s for OME-Zarr microscopy datasets.

    [:octicons-arrow-right-24: Reference](packages/viscy-data.md)

-   :material-graph-outline:{ .lg .middle } __viscy-models__

    ---

    Neural network architectures -  UNet, contrastive encoders, VAEs.

    [:octicons-arrow-right-24: Reference](packages/viscy-models.md)

-   :material-image-auto-adjust:{ .lg .middle } __viscy-transforms__

    ---

    Image transforms tuned for virtual staining microscopy.

    [:octicons-arrow-right-24: Reference](packages/viscy-transforms.md)

-   :material-tools:{ .lg .middle } __viscy-utils__

    ---

    Shared utilities across the other packages and applications.

    [:octicons-arrow-right-24: Reference](packages/viscy-utils.md)

</div>


!!! tip "New here?"

    Start with [Get started](get-started.md) to install VisCy and then browse the [packages overview](packages/index.md) for the live
    version of each package.

!!! info "Interested in contrbuting?"

    See [Contributing](contributing.md) for environment setup, the pull request
    workflow, testing, and building the docs.
