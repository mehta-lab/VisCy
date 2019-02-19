# MicroDL

This is a pipeline for training U-Net models. It consists of three modules:

* Preprocessing: normalization, flatfield correction, masking, tiling
* Training: model creation, loss functions (w/wo masks), metrics, learning rates
* Inference: on full images or on tiles that can be stitched to full images

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

If you want to launch a Jupyter notebook inside your container, you can do so with the following command:
```buildoutcfg
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser
```
Then you can access your notebooks in your browser at:
```buildoutcfg
http://fry:<whatever port you mapped to when starting up docker>
```
You will need to copy/paste the token generated in your Docker container.

### Data Format

To train directly on datasets that have already been split into 2D frames, the dataset
should have the following structure:

```buildoutcfg
dir_name
    |
    |- frames_meta.csv
    |- global_metadata.csv
    |- im_c***_z***_t***_p***.png
    |- im_c***_z***_t***_p***.png
    |- ...
```
The image naming convention is (parenthesis is their name in frames_meta.csv)
* **c** = channel index     (channel_idx)
* **z** = slice index in z stack (slice_idx)
* **t** = timepoint index   (time_idx)
* **p** = position (field of view) index (pos_idx)

If you download your dataset from the imaging database [imagingDB](https://github.com/czbiohub/imagingDB)
you will get your dataset correctly formatted and can directly input that into microDL.
If you don't have your data in the imaging database, write a script that converts your 
your data to image files that adhere to the naming convention above, then run 

```buildoutcfg
python micro_dl/cli/generate_meta.py --input <directory name>
```
That will generate the frames_meta.csv file you will need for data preprocessing.


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

## Training

### Learning Rate
Prior to training, you may want to explore learning rates using the learning rate (LR) finder.
You can see an example on how to run it in 'config_lr_finder.yml'.
The implemented LR finder is based on the [paper by Smith.](https://arxiv.org/abs/1506.01186)
but with the adaptation from [fast.ai](http://www.fast.ai/) to plot loss instead of accuracy
on the y-axis.
The LR finder should be run for a few epochs at most. During those epochs it gradually increase
the learning rate from base_lr to max_lr.
It saves a plot of the results from which you can determine learning
rate bounds from learning rate vs. loss.
Before proceeding to actual training, update base_lr and max_lr to be within the range where
loss starts decreasing to a bit before the loss becomes unstable or start increasing again.
E.g. in the figure below you migth select base_lr = 0.001 and max_lr = 0.006.
![LR Finder](lr_finder_result.png?raw=true "Title")

While training, you can either set the training rate to a value (e.g. from LR finder)
or you can use cyclic learning rates where you set an upper and lower bound for the learning rate.
Cyclical learning rate (CLR) is a custom callback function implemented as in the same paper by Smith
referenced above.
Learning rate is increased then decreased in a repeated triangular
pattern over time. One triangle = one cycle.
Step size is the number of iterations / batches in half a cycle.
The paper recommends a step-size of 2-10 times the number of batches in
an epoch (empirical) i.e step-size = 2-10 epochs.
It might be best to stop training at the end of a cycle when the learning rate is
at minimum value and accuracy/performance potentially peaks.
Initial amplitude is scaled by gamma ** iterations. An example of learning rate with
exponential decay can be seen below.

![LR Finder](CLR.png?raw=true "Title")

### Run Training

Assuming you have a config file that specifies what you would like to train
(see examples config.yml or config_regression.yml), you can start training with the command
```buildoutcfg
python micro_dl/cli/train_script.py --config <config yml file> --gpu <gpu id (default 0)> --gpu_mem_frac <0-1 (default 1> --model_fname <file name if starting from weights>
```

    

## Inference
