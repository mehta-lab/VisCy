"""Model inference related functions"""
from keras import Model

import micro_dl.utils.aux_utils as aux_utils


def load_model(network_config, model_fname, predict=False):
    """Load the model from model_dir

    Due to the lambda layer only model weights are saved and not the model
    config. Hence load_model wouldn't work here!
    :param yaml network_config: a yaml file with all the required parameters
    :param str model_fname: fname with full path of the .hdf5 file with saved
     weights
    :param bool predict: load model for predicting / training. predict skips
     checks on input shape
    :return: Keras.Model instance
    """
    network_config['width'] = None
    network_config['height'] = None
    network_cls = network_config['class']
    if network_cls == 'UNet3D':
        network_config['depth'] = None
    # not ideal as more networks get added
    network_cls = aux_utils.import_object('networks', network_cls)
    network = network_cls(network_config, predict)
    inputs, outputs = network.build_net()
    model = Model(inputs=inputs, outputs=outputs)
    model.load_weights(model_fname)
    return model


def predict_large_image(model, input_image):
    """Predict on an image larger than the one it was trained on

    All networks with U-net like architecture in this repo, use downsampling of
    2, which is only conducive for images with shapes in powers of 2. If
    different, please crop / resize accordingly to avoid shape mismatches.

    :param keras.Model model: Model instance
    :param np.array input_image: as named. expected shape:
     [num_channels, (depth,) height, width] or
     [(depth,) height, width, num_channels]
    :return np.array predicted image: as named. Batch axis removed (and channel
     axis if num_channels=1)
    """
    im_size = input_image.shape
    num_dims = len(im_size)
    assert num_dims in [4, 5], \
        'Invalid image shape: only 4D and 5D inputs - 2D / 3D ' \
        'images with channel and batch dim allowed'

    predicted_image = model.predict(input_image)
    return predicted_image
