from keras import Sequential
from keras.layers import Dense, np

from detectors.single_image_based_detectors.abs_single_image_autoencoder import AbstractSingleImageAD
from detectors.single_image_based_detectors.autoencoder_batch_generator import AutoencoderBatchGenerator
from detectors.anomaly_detector import AnomalyDetector
from utils import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS

INPUT_SHAPE = (IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS,)

class DeepAutoencoder(AbstractSingleImageAD, AnomalyDetector):

    def get_input_shape(self):
        return INPUT_SHAPE

    def __init__(self, name: str, args, hidden_layer_dim=256):
        self.hidden_layer_dim = hidden_layer_dim
        super(DeepAutoencoder, self).__init__(name=name, is_keras_model=True, args=args)

    def _create_keras_model(self, args=None):
        model = Sequential()
        model.add(Dense(self.hidden_layer_dim, activation='relu', input_shape=INPUT_SHAPE))
        model.add(Dense(self.hidden_layer_dim // 2, activation='relu'))
        model.add(Dense(self.hidden_layer_dim // 4, activation='sigmoid'))
        model.add(Dense(self.hidden_layer_dim // 2, activation='relu'))
        model.add(Dense(self.hidden_layer_dim, activation='relu'))
        model.add(Dense(IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS, activation='sigmoid'))
        return model

    def normalize_and_reshape(self, x):
        x = x.astype('float32') / 255.
        x = np.reshape(x, newshape=INPUT_SHAPE)
        return x
