from keras import regularizers, Sequential
from keras.layers import Dense

from detectors.single_image_based_detectors.abs_single_image_autoencoder import AbstractSingleImageAD
from detectors.single_image_based_detectors.autoencoder_batch_generator import AutoencoderBatchGenerator
from detectors.anomaly_detector import AnomalyDetector
from utils import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS

INPUT_SHAPE = (IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS,)


class SimpleAutoencoder(AbstractSingleImageAD, AnomalyDetector):

    def __init__(self, name: str, args):
        super(SimpleAutoencoder, self).__init__(name=name, is_keras_model=True, args=args)

    def get_input_shape(self):
        return INPUT_SHAPE

    def _create_keras_model(self, args=None):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=INPUT_SHAPE, activity_regularizer=regularizers.l1(10e-9)))
        model.add(Dense(IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS, activation='sigmoid'))
        return model

    def normalize_and_reshape(self, x):
        x = x.astype('float32') / 255.
        x = x.reshape(-1, IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS)
        return x
