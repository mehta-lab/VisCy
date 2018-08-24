import keras.backend as K
from keras.callbacks import Callback

import micro_dl.plotting.plot_utils as plot_utils

class LRFinder(Callback):


    def __init__(
            self,
            fig_fname,
            max_epochs=3,
            base_lr=0.0001,
            max_lr=0.1,
    ):
        """
        Learning rate finder. Run for a few epochs at most,
        while gradually increasing learning rate from base_lr to max_lr.
        Saves a plot of the results from which you can determine learning
        rate bounds from learning rate vs. loss.
        Based on Smith's paper: https://arxiv.org/abs/1506.01186
        But with the adaptation from fast.ai (http://www.fast.ai/)
        to plot loss instead of accuracy: https://towardsdatascience.com/
        estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0
        https://github.com/surmenok/keras_lr_finder
        https://github.com/LucasAnders1/LearningRateFinder

        :param fig_fname: Figure file name for saving
        :param max_epochs: Maximum number of epochs
        :param base_lr: Base (minimum) learning rate
        :param max_lr: Maximum learning rate
        """
        super().__init__()

        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.fig_fname = fig_fname
        self.local_lr = base_lr
        self.step_size = 1.
        self.total_steps = 1
        self.iterations = 0
        self.losses = []
        self.lrs = []

    def on_train_begin(self, logs=None):
        """
        Set base learning rate at the beginning of training and get step size
        for learning rate increase

        :param logs: Logging from super class
        """
        logs = logs or {}

        K.set_value(self.model.optimizer.lr, self.base_lr)
        self.total_steps = self.max_epochs * self.params['steps']
        self.step_size = (self.max_lr - self.base_lr) / self.total_steps

    def on_batch_end(self, batch, logs=None):
        """
        Increase learning rate gradually after each batch.

        :param batch: Batch number from Callback super class
        :param logs: Log from super class (required)
        """
        logs = logs or {}

        self.iterations += 1
        if self.iterations >= self.total_steps or self.local_lr >= self.max_lr:
            self.model.stop_training = True
        self.local_lr = self.base_lr + self.iterations * self.step_size
        K.set_value(self.model.optimizer.lr, self.local_lr)
        self.losses.append(logs.get('loss'))
        self.lrs.append(self.local_lr)

    def on_train_end(self, logs=None):
        """
        After finishing the increase from base_lr to max_lr, save plot.

        :param logs: Log from super class
        """
        plot_utils.save_plot(
            x=self.lrs,
            y=self.losses,
            fig_labels=["Learning Rate", "Loss"],
            fig_fname=self.fig_fname,
        )
