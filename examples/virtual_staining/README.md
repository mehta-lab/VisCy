# VisCy usage examples

Examples scripts showcasing the usage of VisCy.

## Virtual staining

### Training

- WIP: DL@MBL notebooks

# Image-to-Image translation using VisCy
- [Image translation Exercise](../demo_dlmbl/solution.py):
Example showing how to use VisCy to train, predict and evaluate the VSCyto2D model. This notebook was developed for the [DL@MBL2024](https://github.com/dlmbl/DL-MBL-2024) course.

- [Virtual staining exercise](./virtual_staining_exercise/solution.py):
Phase to nuclei and plasma mebrane training, prediction and evaluation.
Fluorescence nuclei or membrane to label-free training, prediction and evaluation.

### Inference

- [Inference with VSCyto2D](./VS_model_inference/demo_vscyto2d.py):
2D inference example on 20x A549 cell data. (Phase to nuclei and plasma membrane).
- [Inference with VSCyto3D](./VS_model_inference/demo_vscyto3d.py):
3D inference example on 63x HEK293T cell data. (Phase to nuclei and plasma membrane).
- [Inference VSNeuromast](./VS_model_inference/demo_vsneuromast.py):
3D inference example of 63x zebrafish neuromast data (Phase to nuclei and plasma membrane)

## Notes

To run the examples, execute each individual script, for example:

```sh
python demo_vscyto2d.py
```

These scripts can also be ran interactively in many IDEs as notebooks,
for example in VS Code, PyCharm, and Spyder.
