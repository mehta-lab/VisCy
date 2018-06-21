"""Keras trainer"""
from keras import Model
from keras.models import load_model
from keras import callbacks as keras_callbacks
from keras import optimizers as keras_optimizers
from keras.utils import plot_model
import os
import tensorflow as tf
from time import localtime, strftime
import yaml

from micro_dl.train import learning_rates as custom_learning
from micro_dl.train.model_inference import load_model
from micro_dl.utils.aux_utils import import_class, init_logger
from micro_dl.utils.train_utils import set_keras_session, get_loss, get_metrics


class BaseKerasTrainer:
    """Keras training class"""

    def __init__(self, config, model_dir, train_dataset, val_dataset,
                 model_name=None, gpu_ids=0, gpu_mem_frac=0.95):
        """Init

        Currently only model weights are stored and not the training state.
        Resume training needs to be modified!

        :param dict config: dict read from a config yaml, with parameters for
         dataset, network and trainer
        :param str model_dir: dir with full path to store all training related
         info (model weights, log files, model_graph, etc)
        :param BaseDataSet/DataSetWithMask train_dataset: generator used for
         batching train images
        :param BaseDataSet/DataSetWithMask val_dataset: generator used for
         batching validation images
        :param str model_name: fname of the .hdf5 file with model weights
        :param int/list gpu_ids: gpu to use
        :param float/list gpu_mem_frac: Memory fractions to use corresponding
         to gpu_ids
        """

        self.gpu_ids = gpu_ids
        self.gpu_mem_frac = gpu_mem_frac
        self.verbose = 10
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.model_name = model_name
        self.config = config
        self.epochs = self.config['trainer']['max_epochs']
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.logger = self._init_train_logger()
        if model_name:
            self.resume_training = True
        else:
            self.resume_training = False
        if gpu_ids == -1:
            self.sess = None
        else:
            self.sess = set_keras_session(
                gpu_ids=gpu_ids,
                gpu_mem_frac=gpu_mem_frac)

    def _init_train_logger(self):
        """Initialize logger for training"""

        logger_fname = os.path.join(self.model_dir,
                                    'training.log')
        logger = init_logger('training', logger_fname, self.verbose)
        return logger

    def _get_optimizer(self):
        """Get optimizer from config"""

        opt = self.config['trainer']['optimizer']['name']
        lr = self.config['trainer']['optimizer']['lr']
        try:
            opt_cls = getattr(keras_optimizers, opt)
            return opt_cls(lr=lr)
        except Exception as e:
            self.logger.error('Optimizer not valid: ' + str(e))

    def _get_callbacks(self):
        """Get the callbacks from config"""

        callbacks_config = self.config['trainer']['callbacks']
        callbacks = []
        for cb_dict in callbacks_config:
            cb_cls = getattr(keras_callbacks, cb_dict)
            if cb_dict == 'ModelCheckpoint':
                if self.model_name:
                    model_name = self.model_name
                else:
                    model_name = self.config['network']['class']
                timestamp = strftime("%Y-%m-%d-%H-%M-%S", localtime())
                model_name = '{}_{}.hdf5'.format(model_name, timestamp)
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
                cur_cb = custom_learning.CyclicLearning(
                    base_lr=callbacks_config[cb_dict]['base_lr'],
                    max_lr=callbacks_config[cb_dict]['max_lr'],
                    step_size=callbacks_config[cb_dict]['step_size'],
                    gamma=callbacks_config[cb_dict]['gamma'],
                    scale_mode=callbacks_config[cb_dict]['scale_mode'],
                )
            elif cb_dict == 'TensorBoard':
                log_dir = os.path.join(self.model_dir, 'tensorboard_logs')
                os.makedirs(log_dir, exist_ok=True)
                # https://github.com/keras-team/keras/issues/3358
                # If printing histograms, validation_data must be provided,
                # and cannot be a generator
                cur_cb = cb_cls(
                    log_dir=log_dir,
                    batch_size=self.config['trainer']['batch_size'],
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

    def _init_model(self, loss, optimizer, metrics):
        """Initialize the model from config

        :param loss: instance of keras loss or custom loss
        :param optimizer: instance of an optimizer
        :param metrics: a list of metrics instances
        """

        network_cls = self.config['network']['class']
        network_cls = import_class('networks.unet', network_cls)
        network = network_cls(self.config)
        # assert if network shape matches dataset shape?
        inputs, outputs = network.build_net()
        with tf.device('/gpu:{}'.format(self.gpu_ids)):
            model = Model(inputs=inputs, outputs=outputs)

        plot_model(model,
                   to_file=os.path.join(self.model_dir,'model_graph.png'),
                   show_shapes=True, show_layer_names=True)
        # Lambda layers throw errors when converting to yaml!
        # model_yaml = self.model.to_yaml()
        # with open(os.path.join(self.model_dir, 'model.yml'), 'w') as f:
        #     f.write(model_yaml)
        with open(os.path.join(self.model_dir, 'config.yml'), 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        if 'weighted_loss' in self.config['trainer']:
            n_output_channels = len(self.config['dataset']['target_channels'])
            model.compile(loss=loss(n_output_channels), optimizer=optimizer,
                          metrics=metrics)
        else:
            model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return model

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

        loss_str = self.config['trainer']['loss']
        loss = get_loss(loss_str)
        optimizer = self._get_optimizer()
        metrics_list = self.config['trainer']['metrics']
        metrics = get_metrics(metrics_list)
        callbacks = self._get_callbacks()

        if self.model_name:
            model = load_model(self.config, self.model_name)
            model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
            self.logger.info('Resume model training')
        else:
            os.makedirs(self.model_dir, exist_ok=True)
            model = self._init_model(loss, optimizer, metrics)
            model.summary()
            self.logger.info('Model initialized and compiled')

        try:
            # NUM WORKERS read from yml or find the num of empty cores?
            model.fit_generator(
                generator=self.train_dataset, validation_data=self.val_dataset,
                epochs=self.config['trainer']['max_epochs'],
                callbacks=callbacks, workers=4, verbose=1
            )
        except Exception as e:
            self.logger.error('problems with fit_generator: ' + str(e))
