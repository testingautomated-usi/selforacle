import argparse
import logging
import os

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

import model_factory
import utils_train_self_driving_car
from batch_generator import Generator

np.random.seed(0)
logger = logging.Logger("train_self_driving_car")


def load_data(args):
    """
    Load training data and split it into training and validation set
    """
    tracks = ["track1", "track2", "track3"]
    drive = ["normal", "reverse", "sport_normal", "sport_reverse"]

    x = None
    y = None
    for track in tracks:
        for drive_style in drive:
            try:
                data_df = pd.read_csv(os.path.join(args.data_dir, track, drive_style, 'driving_log.csv'))
                if x is None:
                    x = data_df[['center', 'left', 'right']].values
                    y = data_df['steering'].values
                else:
                    x = np.concatenate((x, data_df[['center', 'left', 'right']].values), axis=0)
                    y = np.concatenate((y, data_df['steering'].values), axis=0)
            except FileNotFoundError:
                continue

    X_train, X_valid, y_train, y_valid = train_test_split(x, y, test_size=args.test_size, random_state=0)

    print("Train dataset: " + str(len(X_train)) + " elements")
    print("Test dataset: " + str(len(X_valid)) + " elements")
    return X_train, X_valid, y_train, y_valid


def build_model(args):
    return model_factory.get_model(args)


def train_model(model, args, X_train, X_valid, y_train, y_valid):
    """
    Train the model
    """
    checkpoint = ModelCheckpoint(args.model + '-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')

    train_generator = Generator(X_train, y_train, True, args)
    validation_generator = Generator(X_valid, y_valid, False, args)

    model.fit_generator(train_generator,
                        validation_data=validation_generator,
                        epochs=args.nb_epoch,
                        use_multiprocessing=False,
                        max_queue_size=10,
                        workers=4,
                        callbacks=[checkpoint],
                        verbose=1)


def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default='../datasets/dataset5/')
    parser.add_argument('-t', help='test size fraction', dest='test_size', type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability', dest='keep_prob', type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs', dest='nb_epoch', type=int, default=500)
    parser.add_argument('-s', help='samples per epoch', dest='samples_per_epoch', type=int, default=100)
    parser.add_argument('-b', help='batch size', dest='batch_size', type=int, default=256)
    parser.add_argument('-o', help='save best models only', dest='save_best_only', type=s2b, default='true')
    parser.add_argument('-l', help='learning rate', dest='learning_rate', type=float, default=1.0e-4)
    parser.add_argument('-sl', help='sequence length', dest='sequence_length', type=int, default=3)
    parser.add_argument('-m', help='model', dest='model', type=str, default="chauffeur")
    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)


    data = utils_train_self_driving_car.load_data(args)
    model = model_factory.get_model(args)
    assert model is not None
    train_model(model, args, *data)


if __name__ == '__main__':
    main()
