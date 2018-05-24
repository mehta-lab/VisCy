"""Model inference related functions"""
import os
from keras import Model

from micro_dl.utils.aux_utils import import_class
from micro_dl.plotting.plot_utils import save_predicted_images


def load_model(config, model_fname):
    """Load the model from model_dir

    Due to the lambda layer only model weights are saved and not the model
    config. Hence load_model wouldn't work here!
    :param yaml config: a yaml file with all the required parameters
    :param str model_fname: fname with full path of the .hdf5 file with saved
     weights
    :return: Keras.Model instance
    """

    network_cls = config['network']['class']
    # not ideal as more networks get added
    network_cls = import_class('networks.unet', network_cls)
    network = network_cls(config)
    inputs, outputs = network.build_net()
    model = Model(inputs=inputs, outputs=outputs)
    model.load_weights(model_fname)
    return model


def predict_and_save_images(config, model_fname, ds_test, model_dir):
    """Run inference on images

    :param yaml config: config used to train the model
    :param str model_fname: fname with full path for the saved model
     (.hdf5)
    :param dataset ds_test: generator for the test set
    :param str model_dir: dir where model results are to be saved
    """

    model = load_model(config, model_fname)
    output_dir = os.path.join(model_dir, 'test_predictions')
    os.makedirs(output_dir, exist_ok=True)
    for batch_idx in range(ds_test.__len__()):
        if 'weighted_loss' in config['trainer']:
            cur_input, cur_target, cur_mask = ds_test.__getitem__(batch_idx)
        else:
            cur_input, cur_target = ds_test.__getitem__(batch_idx)
        pred_batch = model.predict(cur_input)
        save_predicted_images(cur_input, cur_target, pred_batch,
                              output_dir, batch_idx)