import logging

from keras import Sequential
from keras.layers import Flatten, Dropout, Dense, Lambda, Conv2D
from keras.optimizers import Adam

from models.abstract_model_provider import AbstractModelProvider
from utils_train_self_driving_car import INPUT_SHAPE

logger = logging.Logger("Dave2")
NAME = "dave2"


class Dave2(AbstractModelProvider):
    def get_name(self) -> str:
        return NAME

    def get_model(self, args):
        """
        Modified NVIDIA model
        """
        model = Sequential()
        model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE))
        model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
        model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
        model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='elu'))
        model.add(Conv2D(64, (3, 3), activation='elu'))
        model.add(Dropout(args.keep_prob))
        model.add(Flatten())
        model.add(Dense(100, activation='elu'))
        model.add(Dense(50, activation='elu'))
        model.add(Dense(10, activation='elu'))
        model.add(Dense(1))
        model.summary()

        model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))

        return model