from keras import Input, Model
from keras.layers import Dense, Lambda, K

from utils_train_self_driving_car import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS


def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


class VariationalAutoencoder:

    def __init__(self, model_name, intermediate_dim=512, latent_dim=2):
        self.model_name = model_name
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim

    def create_autoencoder(self):
        intermediate_dim = self.intermediate_dim
        latent_dim = self.latent_dim
        input_shape = (IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS,)
        # build encoder model
        inputs = Input(shape=input_shape, name='encoder_input')
        x = Dense(intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)

        # use re-parameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

        # instantiate encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(intermediate_dim, activation='relu')(latent_inputs)
        outputs = Dense(IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS, activation='sigmoid')(x)

        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')

        # instantiate VAE model
        outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs, name='vae_mlp')
        return vae

    def normalize_and_reshape(self, x):
        x = x.astype('float32') / 255.
        x = x.reshape(-1, IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS)
        return x

    def reshape(self, x):
        x = x.reshape(-1, IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS)
        return x
