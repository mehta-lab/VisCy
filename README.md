# MicroDL

This is a pipeline for training U-Net models. It consists of three modudules:

* Preprocessing: normalization, flatfield correction, masking, tiling
* Training: 2D-3D model creation, training with certain losses, metrics, learning rates
* Inference: perform inference on tiles that can be stitched to full images

## Getting Started

### Docker

It is recommended that you run microDL inside a Docker container, especially if you're using shared resources like Fry
or Fry2. microDL comes with two Docker images, one for Python3.6 with CUDA 9 support (which is most likely what
you'll want), and one for Python3.5 with CUDA 8.0 support. You should be in the Docker group on Fry/Fry2, if not you
can request to join. The Python 3.6 image is already built on Fry/Fry2, but if you want to modify it and build your own,
you can do so:
```
docker build -t imaging_docker:gpu_py36_cu90 -f Dockerfile.imaging_docker_py36_cu90 .
```
Now you want to start a Docker container from your image, which is the virtual environment you will run your code in.
```buildoutcfg
nvidia-docker run -it -p <your port>:<exposed port> -v <your dir>:/<dirname inside docker> imaging_docker:gpu_py36_cu90 bash
```
If you look in the Dockerfile, you can see that there are two ports exposed, one is typically used for Jupyter (8888)
and one for Tensorboard (6006). To be able to view these in your browser, you need map the port with the -p argument.
The -v arguments similarly maps directories. You can use multiple -p and -v arguments if you want to map multiple things.
The final 'bash' is to signify that you want to run bash (your usual Unix shell). 

You may need to export pythonpaths inside your Docker container, e.g.:
```buildoutcfg
export PYTHONPATH=$(pwd)
```
for current directory, or whatever other pythonpaths you want to export.

If you want to launch a Jupyter notebook inside your container, you can do so with the following command:
```buildoutcfg
--ip=0.0.0.0 --port=8888 --allow-root --no-browser
```
Then you can access your notebooks in your browser at:
```buildoutcfg
http://fry:<whatever port you mapped to when starting up docker>
```
You will need to copy/paste the token generated in your Docker container.

### Basic CLIs

To run preprocessing on a Lif file, see config_preprocess.yml file
and then run

```buildoutcfg
python micro_dl/input/preprocess_script.py --config micro_dl/config_preprocess.yml
```

To run preprocessing on images, they need to be in the following folder structure

```buildoutcfg
dir_name
    |
    |- timepoint_0
        |
        |- channel_0
            |
            |- image_n0_z0.png
            |- image_n1_z0.png
            |- ...           
        |- channel_1
        |- ...
    |
    |- timepoint_1
        |
        |- channel_0
        |- channel_1
        |- ...
    |
    |- ...
```
That is the same structure that a Lif file will be decomposed into when converting it to
npy arrays, which is the first step in the Lif-file specific CLI preprocess_script.
If your starting point is images readable by OpenCV or numpy (e.g. png, tif, npy, ...)
you can use the following command to preprocess them:

```buildoutcfg
python micro_dl/cli/run_image_preprocessing.py -i <dir_name> -o <output_dir>
--tile_size (optional) --step_size (optinal)
```

To train the model using your preprocessed data, you can modify the followind config file and run:

```buildoutcfg
python micro_dl/train/train_script.py --config micro_dl/config.yml
```

for model inference run:
```buildoutcfg

```

## Requirements

* keras
* tensorflow
* cv2
* Cython
* matplotlib
* natsort
* nose
* numpy
* pandas
* PIMS
* pydot
* scikit-image
* scikit-learn
* scipy
* testfixtures
* python-bioformats (for Lif)
* javabridge (for Lif)


## Preprocessing

The following settings can be adjusted in preprocessing:
* base_output_dir: (str) folder name
* focal_plane_idx: (int) if more than one z-index present (3D image), select oen focal plane (for 2D analysis)
* verbose: (int) Logging verbosity levels: NOTSET:0, DEBUG:10, INFO:20, WARNING:30, ERROR:40, CRITICAL:50
* split_volumes: (bool) split Lif file into 
* splitter_class: LifStackSplitter2D or LifStackSplitter3D
* input_fname: (str) full path to lif file
* correct_flat_field: (bool) perform flatfield correction (2D data only)
* flat_field_class: FlatFieldEstimator2D
* use_masks: (bool) whether to generate binary masks from images
* masks:
    * mask_channels: (list of ints) which channels should be used for masks
    * str_elem_radius: (int) morpological structuring element radius
* tile_stack: (bool) do tiling (recommended)
* tile:
    * channels: (list of ints) specify channel numbers, -1 for all channels
    * tile_size: (list of ints) tile size in pixels for each dimension
    * step_size: (list of ints) step size in pixels for each dimension
    * save_cropped_masks: (bool) save tiled masks
    * min_fraction: (float) minimum fraction of image occupied by foreground in masks
    * hist_clip_limits: (list of ints) lower and upper intensity percentiles for histogram clipping

During preprocessing, a csv file named tiled_images_info.csv will be generated, which
will be used for further processing. The csv contains the following fields for each image tile:

* 'timepoint': the timepoint it came from
* 'channel_num': its channel
* 'sample_num': the image index 
* 'slice_num': the z index in case of 3D data (currently supported in Lif files)
* 'fname': file name
* 'size_x_microns': pixel size in x (microns) (currently supported in Lif)
* 'size_y_microns': pixel size in y (microns) (currently supported in Lif)
* 'size_z_microns': pixel size in z (microns) (currently supported in Lif)


## Modeling


## Inference
