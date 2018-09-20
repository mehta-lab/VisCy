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
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser
```
Then you can access your notebooks in your browser at:
```buildoutcfg
http://fry:<whatever port you mapped to when starting up docker>
```
You will need to copy/paste the token generated in your Docker container.

### Basic CLIs

To train directly on datasets that have already been split into 2D frames, the dataset
should have the following structure:

```buildoutcfg
dir_name
    |
    |- frames_meta.csv
    |- global_metadata.csv
    |- im_c***_t***_p***_z***.png
    |- im_c***_t***_p***_z***.png
    |- ...
```
The image naming convention is (parenthesis is their name in frames_meta.csv)
* **c** = channel index     (channel_idx)
* **t** = timepoint index   (time_idx)
* **p** = position (field of view) index (pos_idx)
* **z** = slice index in z stack (slice_idx)

If you download your dataset from the imaging database [imagingDB](https://github.com/czbiohub/imagingDB)
you will get your dataset correctly formatted and can directly input that into microDL.

If your starting point is images readable by OpenCV or numpy
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


## Preprocessing

The following settings can be adjusted in preprocessing:
* input_dir: (str) Directory where data to be preprocessed is located
* output_dir: (str) folder name where all processed data will be written
* slice_ids: (int/list) Value(s) of z-index to be processed
* verbose: (int) Logging verbosity levels: NOTSET:0, DEBUG:10, INFO:20, WARNING:30, ERROR:40, CRITICAL:50
* correct_flat_field: (bool) perform flatfield correction (2D data only)
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

During preprocessing, a csv file named frames_csv.csv will be generated, which
will be used for further processing. The csv contains the following fields for each image tile:

* 'time_idx': the timepoint it came from
* 'channel_idx': its channel
* 'slice_idx': the z index in case of 3D data
* 'pos_idx': the field of view index
* 'file_name': file name
* 'row_start': starting row for tile (add tile_size for endpoint)
* 'col_idx': start column (add tile_size for endpoint)

## Modeling


## Inference
