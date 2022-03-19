import numpy as np
import tensorflow as tf
# import tensorflow.keras as K
import keras as K
# from tensorflow.keras import backend as Kb
import matplotlib.pyplot as plt
from micro_dl.networks.base_conv_net import BaseConvNet
import micro_dl.utils.aux_utils as aux_utils
print(tf.__version__)
print(K.__version__)

# Hyperparameters
# NUM_LATENT_K = 10                 # Number of codebook entries
# NUM_LATENT_D = 64                 # Dimension of each codebook entries
# BETA = 1.0                        # Weight for the commitment loss
# INPUT_SHAPE = x_train.shape[1:]
# SIZE = None                       # Spatial size of latent embedding
#                                   # will be set dynamically in `build_vqvae
# VQVAE_BATCH_SIZE = 128            # Batch size for training the VQVAE
# VQVAE_NUM_EPOCHS = 20             # Number of epochs
# VQVAE_LEARNING_RATE = 3e-4        # Learning rate
# VQVAE_LAYERS = [16, 32]           # Number of filters for each layer in the encoder

class VectorQuantizer(K.layers.Layer):
    def __init__(self, k, **kwargs):
        super(VectorQuantizer, self).__init__(**kwargs)
        self.k = int(k)

    def build(self, input_shape):
        self.d = int(input_shape[-1])
        rand_init = K.initializers.VarianceScaling(distribution="uniform")
        self.codebook = self.add_weight(shape=(self.k, self.d), initializer=rand_init,
                                        trainable=True, name='codebook')

    def call(self, inputs):
        # Map z_e of shape (b, w,, h, d) to indices in the codebook
        lookup_ = tf.reshape(self.codebook, shape=(1, 1, 1, self.k, self.d))
        z_e = tf.expand_dims(inputs, -2)
        dist = tf.norm(z_e - lookup_, axis=-1)
        k_index = tf.argmin(dist, axis=-1, output_type=tf.dtypes.int64)
        return k_index

    def sample(self, k_index):
        # Map indices array of shape (b, w, h) to actual codebook z_q
        lookup_ = tf.reshape(self.codebook, shape=(1, 1, 1, self.k, self.d))
        k_index_one_hot = tf.one_hot(tf.cast(k_index, dtype='int32'), self.k)
        z_q = lookup_ * k_index_one_hot[..., None]
        z_q = tf.reduce_sum(z_q, axis=-2)
        return z_q

class VQVAE(BaseConvNet):
    def __init__(self, network_config, predict=False):
        super().__init__(network_config, predict)
        req_params = ['num_filters_per_block',
                      'num_convs_per_block',
                      'num_embedding',
                      'num_hidden',
                      'skip_merge_type',
                      'upsampling',
                      'num_target_channels',
                      'residual',
                      'block_sequence']
        if not predict:
            param_check, msg = aux_utils.validate_config(
                network_config,
                req_params,
            )
        if not param_check:
            raise ValueError(msg)
        self.config = network_config
        self.num_filters_per_block = network_config['num_filters_per_block']
        self.num_embedding = network_config['num_embedding']
        self.num_hidden = network_config['num_hidden']
        global SIZE

    @property
    def _get_input_shape(self):
        """Return shape of input"""

        if self.config['data_format'] == 'channels_first':
            shape = (self.config['num_input_channels'],
                     self.config['height'],
                     self.config['width'])
        else:
            shape = (self.config['height'],
                     self.config['width'],
                     self.config['num_input_channels'])
        return shape

    def encoder_pass(self, inputs):
        x = inputs
        for i, filters in enumerate(self.num_filters_per_block):
            x = K.layers.Conv2D(filters=filters, kernel_size=3, padding='SAME', activation='relu',
                                strides=(2, 2), name="conv{}".format(i + 1))(x)
        z_e = K.layers.Conv2D(filters=self.num_hidden, kernel_size=3, padding='SAME', activation=None,
                              strides=(1, 1), name='z_e')(x)
        return z_e

    def decoder_pass(self, inputs):
        y = inputs
        for i, filters in enumerate(self.config['num_filters_per_block']):
            y = K.layers.Conv2DTranspose(filters=filters, kernel_size=4, strides=(2, 2), padding="SAME",
                                         activation='relu', name="convT{}".format(i + 1))(y)
        decoded = K.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=(1, 1),
                                           padding="SAME", activation=None, name='output')(y)
        return decoded

    def build_net(self):

        ## Encoder
        encoder_inputs = K.layers.Input(shape=self._get_input_shape, name='encoder_inputs')
        z_e = self.encoder_pass(encoder_inputs)
        SIZE = int(z_e.get_shape()[1])

        ## Vector Quantization
        vector_quantizer = VectorQuantizer(self.num_embedding, name="vector_quantizer")
        codebook_indices = vector_quantizer(z_e)
        encoder = K.Model(inputs=encoder_inputs, outputs=codebook_indices, name='encoder')

        ## Decoder
        decoder_inputs = K.layers.Input(shape=(SIZE, SIZE, self.num_hidden), name='decoder_inputs')
        decoded = self.decoder_pass(decoder_inputs)
        decoder = K.Model(inputs=decoder_inputs, outputs=decoded, name='decoder')

        ## VQVAE Model (training)
        sampling_layer = K.layers.Lambda(lambda x: vector_quantizer.sample(x), name="sample_from_codebook")
        z_q = sampling_layer(codebook_indices)
        # codes = tf.stack([z_e, z_q], axis=-1)
        codes = K.layers.Lambda(lambda x: tf.stack(x, axis=-1), name='latent_codes')([z_e, z_q])
        straight_through = K.layers.Lambda(lambda x: x[1] + tf.stop_gradient(x[0] - x[1]),
                                           name="straight_through_estimator")
        straight_through_zq = straight_through([z_q, z_e])
        reconstructed = decoder(straight_through_zq)
        return encoder_inputs, [reconstructed, codes]

        # vq_vae = K.Model(inputs=encoder_inputs, outputs=[reconstructed, codes], name='vq-vae')

        # ## VQVAE model (inference)
        # codebook_indices = K.layers.Input(shape=(SIZE, SIZE), name='discrete_codes', dtype=tf.int32)
        # z_q = sampling_layer(codebook_indices)
        # generated = decoder(z_q)
        # vq_vae_sampler = K.Model(inputs=codebook_indices, outputs=generated, name='vq-vae-sampler')
        #
        # ## Transition from codebook indices to model (for training the prior later)
        # indices = K.layers.Input(shape=(SIZE, SIZE), name='codes_sampler_inputs', dtype='int32')
        # z_q = sampling_layer(indices)
        # codes_sampler = K.Model(inputs=indices, outputs=z_q, name="codes_sampler")
        #
        # ## Getter to easily access the codebook for vizualisation
        # indices = K.layers.Input(shape=(), dtype='int32')
        # vector_model = K.Model(inputs=indices, outputs=vector_quantizer.sample(indices[:, None, None]),
        #                        name='get_codebook')
        #
        # def get_vq_vae_codebook():
        #     codebook = vector_model.predict(np.arange(self.num_embedding))
        #     codebook = np.reshape(codebook, (self.num_embedding, self.num_hidden))
        #     return codebook
        #
        # return vq_vae, vq_vae_sampler, encoder, decoder, codes_sampler, get_vq_vae_codebook

