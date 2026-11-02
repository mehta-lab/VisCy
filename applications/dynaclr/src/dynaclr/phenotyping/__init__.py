"""Cell segmentation, nuclear membership and cell-specific fluorescence QC.

Heavy image/model dependencies are imported by the relevant adapter, so config
inspection and matrix dry runs do not initialize CUDA or download checkpoints.
"""
