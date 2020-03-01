# -----------------------------------------------------------------------------------------
#  Challenge #2 -  epoch_model.py - Model Structure
# -----------------------------------------------------------------------------------------

'''
build_cnn contains the final model structure for this competition
I also experimented with transfer learning with Inception V3
Original By: dolaameng Revd: cgundling
'''

import logging

from keras import Sequential
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dropout, Dense, K, Lambda
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam

from models.abstract_model_provider import AbstractModelProvider

logger = logging.Logger("Epoch")
NAME = "epoch"


class Epoch(AbstractModelProvider):
    def get_name(self) -> str:
        return NAME

    def get_model(self, args):
        # def build_cnn(image_size=None,weights_path=None):
        # image_size = image_size or (128, 128)
        # if K.image_dim_ordering() == 'th':
        #     input_shape = (3,) + image_size
        # else:
        #     input_shape = image_size + (3, )

        img_input = Input(utils.INPUT_SHAPE)

        model = Sequential()
        model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=utils.INPUT_SHAPE))

        model.add(Convolution2D(32, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(.5))
        model.add(Dense(1))
        model.summary()

        # model = Model(inputs=img_input, outputs=y)
        model.compile(loss='mse', optimizer=Adam(lr=args.learning_rate))

        # if weights_path:
        #     model.load_weights(weights_path)

        return model


def build_InceptionV3(image_size=None, weights_path=None):
    image_size = image_size or (299, 299)
    if K.image_dim_ordering() == 'th':
        input_shape = (3,) + image_size
    else:
        input_shape = image_size + (3,)
    bottleneck_model = InceptionV3(weights='imagenet', include_top=False,
                                   input_tensor=Input(input_shape))
    for layer in bottleneck_model.layers:
        layer.trainable = False

    x = bottleneck_model.input
    y = bottleneck_model.output
    # There are different ways to handle the bottleneck output
    y = GlobalAveragePooling2D()(x)
    # y = AveragePooling2D((8, 8), strides=(8, 8))(x)
    # y = Flatten()(y)
    # y = BatchNormalization()(y)
    y = Dense(1024, activation='relu')(y)
    y = Dropout(.5)(y)
    y = Dense(1)(y)

    model = Model(input=x, output=y)
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    return model
