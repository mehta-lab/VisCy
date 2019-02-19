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
![LR Finder](../lr_finder_result.png?raw=true "Title")

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

![LR Finder](../CLR.png?raw=true "Title")

### Run Training

Assuming you have a config file that specifies what you would like to train
(see examples config.yml or config_regression.yml), you can start training with the command
```buildoutcfg
python micro_dl/cli/train_script.py --config <config yml file> --gpu <gpu id (default 0)> --gpu_mem_frac <0-1 (default 1> --model_fname <file name if starting from weights>
```
