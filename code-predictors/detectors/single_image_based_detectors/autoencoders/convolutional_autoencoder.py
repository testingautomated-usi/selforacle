from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, np, BatchNormalization, Activation

from detectors.single_image_based_detectors.abs_single_image_autoencoder import AbstractSingleImageAD
from detectors.single_image_based_detectors.autoencoder_batch_generator import AutoencoderBatchGenerator
from detectors.anomaly_detector import AnomalyDetector
from utils import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS

INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


class ConvolutionalAutoencoder(AbstractSingleImageAD, AnomalyDetector):

    def __init__(self, name: str, args):
        super(ConvolutionalAutoencoder, self).__init__(name=name, is_keras_model=True, args=args)

    def get_input_shape(self):
        return INPUT_SHAPE

    def _create_keras_model(self, args=None):
        input_img = Input(shape=INPUT_SHAPE)

        x = Conv2D(64, (3, 3), padding='same')(input_img)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(16, (3, 3), padding='same')(encoded)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(3, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        decoded = Activation('sigmoid')(x)

        autoencoder = Model(input_img, decoded)
        return autoencoder

    def normalize_and_reshape(self, x):
        x = x.astype('float32') / 255.
        x = np.reshape(x, newshape=INPUT_SHAPE)  # CNN needs depth.
        return x
