"""Keras trainer"""
from keras import callbacks as keras_callbacks
from keras import optimizers as keras_optimizers
import os
import time


import micro_dl.train.learning_rates as custom_learning
from micro_dl.train.losses import masked_loss
import micro_dl.train.lr_finder as lr_finder
from micro_dl.utils.aux_utils import init_logger
from micro_dl.utils.train_utils import get_loss, get_metrics


class BaseKerasTrainer:
    """Keras training class"""

    def __init__(self, sess, train_config, train_dataset, val_dataset,
                 model, num_target_channels, gpu_ids=0, gpu_mem_frac=0.95):
        """Init

        Currently only model weights are stored and not the training state.
        Resume training needs to be modified!

        :param tf.Session sess: keras session
        :param dict train_config: dict read from a config yaml, with parameters
         for training
        :param BaseDataSet/DataSetWithMask train_dataset: generator used for
         batching train images
        :param BaseDataSet/DataSetWithMask val_dataset: generator used for
         batching validation images
        :param keras.Model model: network instantiated from class but not
         compiled
        :param int num_target_channels: number of channels in target, needed
         for splitting y_true and mask in case of masked_loss
        :param int/list gpu_ids: gpu to use
        :param float/list gpu_mem_frac: Memory fractions to use corresponding
         to gpu_ids
        """

        self.sess = sess
        self.gpu_ids = gpu_ids
        self.gpu_mem_frac = gpu_mem_frac
        self.verbose = 10

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model = model

        self.config = train_config
        os.makedirs(train_config['model_dir'], exist_ok=True)
        self.model_dir = train_config['model_dir']
        self.epochs = train_config['max_epochs']
        self.num_target_channels = num_target_channels
        self.logger = self._init_train_logger()

        workers = 4
        if 'workers' in train_config:
            workers = train_config['workers']
            assert isinstance(workers, int) and workers > 0, \
                'workers must be a positive integer'
        self.workers = workers

        self.resume_training = False
        if 'resume' in train_config and train_config['resume']:
            self.resume_training = True

    def _init_train_logger(self):
        """Initialize logger for training"""

        logger_fname = os.path.join(self.config['model_dir'], 'training.log')
        logger = init_logger('training', logger_fname, self.verbose)
        return logger

    def _get_optimizer(self):
        """Get optimizer from config"""

        opt = self.config['optimizer']['name']
        lr = self.config['optimizer']['lr']
        try:
            opt_cls = getattr(keras_optimizers, opt)
            return opt_cls(lr=lr)
        except Exception as e:
            self.logger.error('Optimizer not valid: ' + str(e))

    def _get_callbacks(self):
        """Get the callbacks from config"""

        callbacks_config = self.config['callbacks']
        callbacks = []
        for cb_dict in callbacks_config:
            cb_cls = getattr(keras_callbacks, cb_dict)
            if cb_dict == 'ModelCheckpoint':
                if callbacks_config[cb_dict]['save_best_only']:
                    assert callbacks_config[cb_dict]['monitor'] == 'val_loss',\
                        'cannot checkpoint best_model if monitor' \
                        'is not val_loss'
                timestamp = time.strftime("%Y-%m-%d-%H-%M-%S",
                                          time.localtime())
                model_name = '{}_{}.hdf5'.format('Model', timestamp)
                filepath = os.path.join(self.model_dir, model_name)
                # https://github.com/keras-team/keras/issues/8343
                # Lambda layer: keras can't make a deepcopy of the layer
                # configuration because there is a tensor in it! LAMBDA :-(
                # save_weights_only
                cur_cb = cb_cls(
                    filepath=filepath,
                    monitor=callbacks_config[cb_dict]['monitor'],
                    mode=callbacks_config[cb_dict]['mode'],
                    save_best_only=callbacks_config[cb_dict]['save_best_only'],
                    save_weights_only=True,
                    verbose=callbacks_config[cb_dict]['verbose']
                )
            elif cb_dict == 'EarlyStopping':
                cur_cb = cb_cls(
                    mode=callbacks_config[cb_dict]['mode'],
                    monitor=callbacks_config[cb_dict]['monitor'],
                    patience=callbacks_config[cb_dict]['patience'],
                    verbose=callbacks_config[cb_dict]['verbose']
                )
            elif cb_dict == 'LearningRateScheduler':
                # Learning rate scheduler should be used either prior to
                # training using LR finder, or for CLR during training
                if callbacks_config[cb_dict]['lr_find']:
                    if 'fig_fname' in callbacks_config[cb_dict]:
                        fig_fname = callbacks_config[cb_dict]['fig_fname']
                    else:
                        # Save figure in model dir with default name
                        fig_fname = os.path.join(self.model_dir,
                                                'lr_finder_result.png')
                    cur_cb = lr_finder.LRFinder(
                        base_lr=callbacks_config[cb_dict]['base_lr'],
                        max_lr=callbacks_config[cb_dict]['max_lr'],
                        max_epochs=callbacks_config[cb_dict]['max_epochs'],
                        fig_fname=fig_fname,
                    )
                else:
                    scale_mode = "cycle"
                    if 'scale_mode' in callbacks_config[cb_dict]:
                        scale_mode = callbacks_config[cb_dict]['scale_mode']
                    cur_cb = custom_learning.CyclicLearning(
                        base_lr=callbacks_config[cb_dict]['base_lr'],
                        max_lr=callbacks_config[cb_dict]['max_lr'],
                        step_size=callbacks_config[cb_dict]['step_size'],
                        gamma=callbacks_config[cb_dict]['gamma'],
                        scale_mode=scale_mode,
                    )
            elif cb_dict == 'TensorBoard':
                log_dir = os.path.join(self.model_dir, 'tensorboard_logs')
                os.makedirs(log_dir, exist_ok=True)
                # https://github.com/keras-team/keras/issues/3358
                # If printing histograms, validation_data must be provided,
                # and cannot be a generator
                cur_cb = cb_cls(
                    log_dir=log_dir,
                    batch_size=self.config['batch_size'],
                    histogram_freq=0, write_graph=True
                )
            else:
                cur_cb = None
            callbacks.append(cur_cb)

        csv_logger = keras_callbacks.CSVLogger(
            filename=os.path.join(self.model_dir, 'history.csv'),
            append=self.resume_training
        )
        callbacks.append(csv_logger)
        return callbacks

    def _compile_model(self, loss, optimizer, metrics):
        """Initialize the model from config

        :param loss: instance of keras loss or custom loss
        :param optimizer: instance of an optimizer
        :param metrics: a list of metrics instances
        """
        loss_is_masked = False
        if 'masked_loss' in self.config:
            loss_is_masked = self.config["masked_loss"]
        if loss_is_masked:
            if metrics is not None:
                masked_metrics = [metric(self.num_target_channels)
                                  for metric in metrics]
            else:
                masked_metrics = metrics
            self.model.compile(loss=masked_loss(loss,
                                                self.num_target_channels),
                               optimizer=optimizer,
                               metrics=masked_metrics)
        else:
            self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def train(self):
        """Train the model

        https://stackoverflow.com/questions/44747288/keras-sample-weight-array-error
        https://gist.github.com/andreimouraviev/2642384705034da92d6954dd9993fb4d
        https://github.com/keras-team/keras/issues/2115

        Suggested: modify generator to return a tuple with (input, output,
        sample_weights) and use sample_weight_mode=temporal. This doesn't fit
        the case for dynamic weighting (i.e. weights change with input image)
        Use model.fit instead of fit_generator as couldn't find how sample
        weights are passed from generator to fit_generator / fit.

        FOUND A HACKY WAY TO PASS DYNAMIC WEIGHTS TO LOSS FUNCTION IN KERAS!
        https://groups.google.com/forum/#!searchin/keras-users/pass$20custom$20loss$20|sort:date/keras-users/ue1S8uAPDKU/x2ml5J7YBwAJ
        """

        loss_str = self.config['loss']
        loss = get_loss(loss_str)
        optimizer = self._get_optimizer()
        callbacks = self._get_callbacks()
        if 'metrics' in self.config:
            metrics_list = self.config['metrics']
            metrics = get_metrics(metrics_list)
        else:
            metrics = None

        self._compile_model(loss, optimizer, metrics)
        self.model.summary()
        self.logger.info('Model compiled')
        steps_per_epoch = self.train_dataset.get_steps_per_epoch()
        self.logger.info("Steps per epoch: {}".format(steps_per_epoch))

        try:
            time_start = time.time()
            # NUM WORKERS read from yml or find the num of empty cores?
            self.model.fit_generator(generator=self.train_dataset,
                                     validation_data=self.val_dataset,
                                     epochs=self.epochs,
                                     steps_per_epoch=steps_per_epoch,
                                     callbacks=callbacks,
                                     workers=self.workers,
                                     verbose=1)
            time_el = time.time() - time_start
            self.logger.info("Training time: {}".format(time_el))
        except Exception as e:
            self.logger.error('problems with fit_generator: ' + str(e))
            raise
