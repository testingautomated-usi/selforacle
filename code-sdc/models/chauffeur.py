import logging

from keras import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dropout, Dense, SpatialDropout2D, K
from keras.optimizers import SGD
from keras.regularizers import l2

from utils_train_self_driving_car import INPUT_SHAPE, rmse
from models.abstract_model_provider import AbstractModelProvider

logger = logging.Logger("Chauffeur")
NAME = "chauffeur"


# Note: For chauffeur you still have to change the following in the drive method
# (not yet done automatically and im not working on it as it does not look like we're going to use chauffeur')
# def rmse(y_true, y_pred):
#     '''Calculates RMSE
#     '''
#     return K.sqrt(K.mean(K.square(y_pred - y_true)))
#
#
# model = load_model(filepath=args.model, custom_objects={"rmse": rmse}, compile=True)


class Chauffeur(AbstractModelProvider):
    def get_name(self) -> str:
        return NAME

    def get_model(self, args):
        logger.warning("We are currently still ignoring the args settings (e.g. args.learning_rate) in chauffeur")

        # Taken from https://github.com/udacity/self-driving-car/blob/master/steering-models/community-models/chauffeur/models.py
        use_adadelta = True
        learning_rate=0.01
        W_l2=0.0001
        input_shape = INPUT_SHAPE # Original Chauffeur uses input_shape=(120, 320, 3)

        model = Sequential()
        model.add(Convolution2D(16, 5, 5,
                                input_shape=input_shape,
                                init= "he_normal",
                                activation='relu',
                                border_mode='same'))
        model.add(SpatialDropout2D(0.1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(20, 5, 5,
            init= "he_normal",
            activation='relu',
            border_mode='same'))
        model.add(SpatialDropout2D(0.1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(40, 3, 3,
            init= "he_normal",
            activation='relu',
            border_mode='same'))
        model.add(SpatialDropout2D(0.1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(60, 3, 3,
            init= "he_normal",
            activation='relu',
            border_mode='same'))
        model.add(SpatialDropout2D(0.1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(80, 2, 2,
            init= "he_normal",
            activation='relu',
            border_mode='same'))
        model.add(SpatialDropout2D(0.1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(128, 2, 2,
            init= "he_normal",
            activation='relu',
            border_mode='same'))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(
            output_dim=1,
            init='he_normal',
            W_regularizer=l2(W_l2)))

        optimizer = ('adadelta' if use_adadelta
                     else SGD(lr=learning_rate, momentum=0.9))

        model.compile(
                loss='mean_squared_error',
                optimizer=optimizer,
                metrics=[rmse])
        return model


