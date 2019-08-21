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
python micro_dl/cli/train_script.py --config <config yml file> --gpu <gpu id> --gpu_mem_frac <memory fraction>
```

where the parameters are defined as follows:
* **config** (yaml file): Configuration file, see below.
* **gpu** (int): ID number of if you'd like to specify which GPU you'd like to run on. If you don't
specify a GPU then the GPU with the largest amount of available memory will be selected for you.
* **gpu_mem_fraction** (float): You can specify what fraction of total GPU memory you'd like to utilize.
If there's not enough memory available on the GPU, and AssertionError will be raised.
If memory fraction is unspecified, all memory currently available on the GPU will automatically
be allocated for you.
## Config File Settings

There are three main blocks you can configure settings for in this module: dataset, trainer and network. 

* **dataset** is where you set things like where is your data, how to split your data and which channels to model.
* **trainer** sets your model directory, callbacks and optimizer.
* **network** sets network parameters like which model class, how to configure blocks, dropouts, activations etc.

Below is a more detailed list of possible config settings. You can see some example configs in the micro_dl directory.

* dataset:
    * data_dir (str): Full path to where your preprocessed or tiled dataset is located
    * input_channels (list): List of integers indicating which channel indices to be used as inputs
    * target_channels (list): List of integers indicating which channel indices to be used as targets
    * train_fraction (float): What fraction of total data will be used in each epoch (0, 1]
    * split_by_column (idx): Which index the dataset split should be done over (pos_idx, channel_idx, time_idx, slice_idx)
    * split_ratio:
        train (float): Fraction of all data to be used for training. train, val and test must sum to 1.
        val (float): Fraction of total data to be used for validation.
        test (float): Fraction of total data to be used for testing.
* verbose (int): Verbosity (default 10)
* trainer:
    * model_dir (str): Directory where model weights, graph and tensorboard log will be written
    * batch_size (int): Mini-batch gradient descent batch size
    * max_epochs (int): Maximum number of epochs to run
    * metrics: Either Keras metrics, or custom metrics in micro_dl/train/metrics.py
    * loss: Either Keras loss, or custom loss in micro_dl/train/losses.py
    * callbacks:
        * EarlyStopping:
            mode: min
            monitor: val_loss
            patience: 50
            verbose: True
        * ModelCheckpoint:
            mode: min
            monitor: val_loss
            save_best_only: True
            verbose: True
        * TensorBoard:
            histogram_freq: 0
            verbose: True
    * optimizer:
        * lr (float): Learning rate
        * name: Any Keras optimizer
* network:
    * class: E.g. UNet2D, UNet3D, UNetStackTo2D, Image2DToVectorNet
    * num_input_channels (int): Number of input channels
    * num_target_channels (int): Number of target channels
    * data_format: 'channels_first'
    * height (int): Tile height
    * width (int): Tile width
    * depth (int): Tile depth (z)
    * batch_norm (bool): Wether to use batch norm
    * pooling_type: E.g. max, average
    * filter_size (int): Filter size. Must be uneven integer
    * activation:
        type: relu
    * dropout (float): Dropout fraction [0, 1]
    * num_filters_per_block (list): List of integers specifying number of filters in each layer
    * num_convs_per_block (int): Number of convolutions per block
    * block_sequence: conv-activation-bn
    * skip_merge_type: Options are concat or add
    * upsampling: Options are bilinear, nearest_neighbor
    * residual (bool): Use residual connections
    * final_activation: Keras activations, e.g. linear, sigmoid, ...
