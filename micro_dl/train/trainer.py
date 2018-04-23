"""Keras trainer"""
from keras import Model
from keras.models import load_model
from keras import callbacks as keras_callbacks
from keras import losses as keras_losses
from keras import metrics as keras_metrics
from keras import optimizers as keras_optimizers
from keras.utils import plot_model
import logging
import os
import tensorflow as tf
from time import localtime, strftime
import yaml

from micro_dl.utils.aux_utils import import_class
from micro_dl.utils.train_utils import set_keras_session

class BaseKerasTrainer:
    """Keras training class"""

    def __init__(self, config, model_dir, train_dataset, val_dataset,
                 model_name=None, gpu_ids=0, gpu_mem_frac=0.9):
        """Init

        :param yaml config:
        """

        self.gpu_ids = gpu_ids
        self.gpu_mem_frac = gpu_mem_frac
        self.verbose = 10
        self.model_dir = model_dir
        self.model_name = model_name
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.logger = self._init_train_logger()
        if model_name:
            self.resume_training = True
        else:
            self.resume_training = False
        self.sess = set_keras_session(gpu_ids=gpu_ids,
                                      gpu_mem_frac=gpu_mem_frac)

    def _init_train_logger(self):
        """Initialize logger for training"""

        logger = logging.getLogger('training')
        logger.setLevel(self.verbose)
        logger.propagate = False

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(self.verbose)
        logger.addHandler(stream_handler)

        logger_fname = os.path.join(self.model_dir,
                                    'training.log')
        file_handler = logging.FileHandler(logger_fname)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(self.verbose)
        logger.addHandler(file_handler)
        return logger

    def _get_loss(self):
        """Get loss type from config"""

        loss = self.config['trainer']['loss']
        if hasattr(keras_losses, loss):
            loss_cls = getattr(keras_losses, loss)
            return loss_cls
        loss_cls = import_class('train.objectives', loss)
        return loss_cls

    def _get_optimizer(self):
        """Get optimizer from config"""

        opt = self.config['trainer']['optimizer']['name']
        lr = self.config['trainer']['optimizer']['lr']
        try:
            opt_cls = getattr(keras_optimizers, opt)
            return opt_cls(lr=lr)
        except Exception as e:
            self.logger.error('Optimizer not valid: ' + str(e))

    def _get_metrics(self):
        """Get the metrics from config"""
 
        metrics = self.config['trainer']['metrics']
        if isinstance(metrics, list):
            metrics_cls = []
            metrics = self.config['trainer']['metrics']
            for m in metrics:
                cur_metric_cls = getattr(keras_metrics, m)
                metrics_cls.append(cur_metric_cls)
            return metrics_cls
        else:
            return [getattr(keras_metrics, metrics)]

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
                cur_cb = cb_cls(
                    filepath=filepath,
                    monitor=callbacks_config[cb_dict]['monitor'],
                    mode=callbacks_config[cb_dict]['mode'],
                    save_best_only=callbacks_config[cb_dict]['save_best_only'],
                    save_weights_only=True,
                    verbose=callbacks_config[cb_dict]['verbose'])
            elif cb_dict == 'EarlyStopping':
                cur_cb = cb_cls(mode=callbacks_config[cb_dict]['mode'],
                                monitor=callbacks_config[cb_dict]['monitor'],
                                patience=callbacks_config[cb_dict]['patience'],
                                verbose=callbacks_config[cb_dict]['verbose'])
            elif cb_dict == 'TensorBoard':
                log_dir = os.path.join(self.model_dir, 'tensorboard_logs')
                os.makedirs(log_dir, exist_ok=True)
                # If printing histograms, validation_data must be provided,
                # and cannot be a generator
                cur_cb = cb_cls(
                    log_dir=log_dir,
                    batch_size=self.config['trainer']['batch_size'],
                    histogram_freq=0, write_graph=True)
            else:
                cur_cb = None
            callbacks.append(cur_cb)

        csv_logger = keras_callbacks.CSVLogger(
            filename=os.path.join(self.model_dir, 'history.csv'),
            append=self.resume_training
        )
        callbacks.append(csv_logger)
        return callbacks

    def _load_model(self):
        """Load the model from model_dir"""

        # find a way to get the last saved model and resume
        self.model = load_model()

    def _init_model(self):
        """Initialize the model from config"""

        network_cls = self.config['network']['class']
        # NEED A BETTER WAY TO IMPORT NETWORK
        network_cls = import_class('networks.unet', network_cls)
        self.network = network_cls(self.config)
        self.inputs, self.outputs = self.network.build_net()
        with tf.device('/gpu:{}'.format(self.gpu_ids)):
            self.model = Model(inputs=self.inputs, outputs=self.outputs)

        plot_model(self.model,
                   to_file=os.path.join(self.model_dir,'model_graph.png'),
                   show_shapes=True, show_layer_names=True)
        # Lambda layers throw errors when converting to yaml!
        # model_yaml = self.model.to_yaml()
        # with open(os.path.join(self.model_dir, 'model.yml'), 'w') as f:
        #     f.write(model_yaml)
        with open(os.path.join(self.model_dir, 'config.yml'), 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

        self.model.compile(loss=self.loss, optimizer=self.optimizer,
                           metrics=self.metrics)

    def train(self):
        """Train the model"""

        self.loss = self._get_loss()
        self.optimizer = self._get_optimizer()
        self.metrics = self._get_metrics()
        self.callbacks = self._get_callbacks()

        if self.model_name:
            self.model = self._load_model()
            self.logger.info('Resume model training from ....')
        else:
            os.makedirs(self.model_dir, exist_ok=True)
            self._init_model()
            self.model.summary()
            self.logger.info('Model initialized and compiled')

        try:
            # NUM WORKERS read from yml or find the num of empty cores?
            self.model.fit_generator(
                generator=self.train_dataset,
                validation_data=self.val_dataset,
                epochs=self.config['trainer']['max_epochs'],
                callbacks=self.callbacks, workers=4, verbose=1
            )
        except Exception as e:
            self.logger.error('problems with fit_generator: ' + str(e))

