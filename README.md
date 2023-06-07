# viscy

viscy is a machine learning toolkit to solve computer vision problems
in high-throughput imaging of cells.

## Predicting sub-cellular structure

Training a model for the segmentation of sub-cellular landmarks
such as nuclei and membrane
directly can require laborious manual annotation.
We use fluorescent markers as a proxy of human-annotated masks
and turn this instance segmentation problem into
an image-to-image translation (I2I) problem.

viscy features an end-to-end pipeline to design, train and evaluate
I2I models in a declarative manner.
It supports 2D, 2.5D (3D encoder, 2D decoder) and 3D U-Nets,
as well as 3D networks with anisotropic filters.
