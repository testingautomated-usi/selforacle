import logging
import os
import shutil

import keras
import numpy
import pandas as pd
from PIL import Image
from keras import Sequential
from keras.layers import ConvLSTM2D, BatchNormalization, Conv3D
from sklearn.metrics import mean_squared_error

import utils
import utils_logging
from detectors.anomaly_detector import AnomalyDetector
from detectors.img_sequence_cnnlstm.lstm_img_batch_generator import LstmImgBatchGenerator, FRAME_INTERVAL
from utils import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS

INPUT_SHAPE = (None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

logger = logging.Logger("CnnLstmImg")
utils_logging.log_info(logger)


class CnnLstmImg(AnomalyDetector):

    def get_input_shape(self):
        return INPUT_SHAPE

    def __init__(self, name: str, args):
        super(CnnLstmImg, self).__init__(name=name, is_keras_model=True, args=args)

    def get_batch_generator(self, x, y, data_dir: str):
        return LstmImgBatchGenerator(x_paths_to_pictures=x,
                                     y_paths_to_pictures=y,
                                     sequence_length=self.args.sequence_length,
                                     gray_scale=self.args.gray_scale,
                                     data_dir=data_dir)

    def _compile(self):
        model = self._get_keras_model()
        model.compile(loss='mean_squared_error', optimizer='adadelta')

    def _create_keras_model(self, args=None) -> keras.Model:
        seq = Sequential()
        seq.add(ConvLSTM2D(filters=40,
                           kernel_size=(1, 1),
                           input_shape=INPUT_SHAPE,
                           padding='same',
                           return_sequences=True,
                           activation='relu'))
        seq.add(BatchNormalization())
        seq.add(ConvLSTM2D(filters=40,
                           kernel_size=(3, 3),
                           input_shape=INPUT_SHAPE,
                           padding='same',
                           return_sequences=True,
                           activation='relu'))
        seq.add(BatchNormalization())
        seq.add(Conv3D(filters=IMAGE_CHANNELS,
                       kernel_size=(1, 1, 1),
                       activation='sigmoid',
                       padding='same',
                       data_format='channels_last'))
        return seq

    def normalize_and_reshape(self, x):
        print("don't use this here")
        exit(1)

    def load_img_paths(self, data_dir: str, restrict_size: bool, eval_data_mode: bool):

        x = None
        frame_ids = None  # Only used in eval data mode
        are_crashes = None  # Only used in eval data mode

        if eval_data_mode:
            data_df = pd.read_csv(os.path.join(data_dir, 'driving_log.csv'))
            x = data_df['center'].values
            frame_ids = data_df['FrameId'].values
            are_crashes = data_df['Crashed'].values

        else:
            tracks = ["track1", "track2", "track3"]
            drive = ["normal", "reverse"]

            for track in tracks:
                for drive_style in drive:
                    data_df = pd.read_csv(os.path.join(data_dir, track, drive_style, 'driving_log.csv'))
                    if x is None:
                        x = data_df['center'].values
                    else:
                        x = numpy.concatenate((x, data_df['center'].values), axis=0)

        assert len(x) > 0

        print("Read %d samples" % len(x))

        sequence_length = self.args.sequence_length

        print("Creating sequences of length %d" % sequence_length)

        x_train = []
        y_train = []
        frame_ids_seqences = []
        are_crashes_sequences = []
        number_of_frames = sequence_length * FRAME_INTERVAL
        for index in range(len(x) - number_of_frames - FRAME_INTERVAL - 1):
            seq_x = x[index: index + number_of_frames: FRAME_INTERVAL]
            seq_y = x[index + FRAME_INTERVAL: index + FRAME_INTERVAL + number_of_frames: FRAME_INTERVAL]
            x_train.append(seq_x)
            y_train.append(seq_y)
            if eval_data_mode:
                frame_id = frame_ids[index + FRAME_INTERVAL + number_of_frames]
                is_crash = are_crashes[index + FRAME_INTERVAL + number_of_frames]
                frame_ids_seqences.append(frame_id)
                are_crashes_sequences.append(is_crash)

        assert x_train[0][1] == y_train[0][0]

        print("Created %d sequences" % len(x_train))

        x_train = numpy.array(x_train)
        y_train = numpy.array(y_train)

        if restrict_size and len(x_train) > self.args.train_abs_size != -1 and not eval_data_mode:
            shuffle_seed = numpy.random.randint(low=1)
            numpy.random.seed(shuffle_seed)
            numpy.random.shuffle(x_train)
            numpy.random.seed(shuffle_seed)
            numpy.random.shuffle(y_train)
            x_train = x_train[:self.args.train_abs_size]
            y_train = y_train[:self.args.train_abs_size]

        assert x_train[0][1] == y_train[0][0]
        if eval_data_mode:
            return x_train, y_train, frame_ids_seqences, are_crashes_sequences
        else:
            print("Train dataset: " + str(len(x)) + " elements")
            return x_train, y_train

    # Don't delete, even if unused. Helper method for manual review and to generate images for slides ...
    def evaluate_optically_img_loss(self, trained: keras.Model, x_test: numpy.array, y_test: numpy.array, data_dir: str,
                                    args):
        """
        For manually checking the predictions (writes a couple of predicted/actual image pairs to the file system)
        :param trained:
        :param x_test:
        :param y_test:
        :param args:
        :return:
        """
        sequence_generator = LstmImgBatchGenerator(x_paths_to_pictures=x_test,
                                                   y_paths_to_pictures=y_test,
                                                   sequence_length=self.args.sequence_length,
                                                   gray_scale=self.args.gray_scale,
                                                   data_dir=data_dir)
        for index in range(10):
            sequence_number = index * 100
            sequence, next_img = sequence_generator.get_single_sequence(sequence_number)
            predicted = trained.predict(sequence)
            predicted = predicted[0][args.sequence_length - 1]
            if not args.gray_scale:
                next_img = next_img * 255
                next_img = next_img.astype('uint8')
                predicted = predicted * 255
                predicted = predicted.astype('uint8')

            reshaped = predicted.reshape(utils.IMAGE_HEIGHT, utils.IMAGE_WIDTH, utils.IMAGE_CHANNELS)

            folder_path = '../temp_generated_imgs/' + str(index) + "/"
            shutil.rmtree(folder_path)
            os.mkdir(folder_path)
            predicted_img = Image.fromarray(reshaped, "RGB")
            predicted_img.save(folder_path + 'predicted.png')
            real_img = Image.fromarray(next_img, "RGB")
            real_img.save(folder_path + 'real.png')

    def _post_training(self, keras_model, x, y, data_dir, args):
        self.evaluate_optically_img_loss(keras_model, x, y, data_dir, args=args)

    def _distance_metrics(self, label: numpy.array, prediction: numpy.array):
        # Use euclid as default for evaluation
        return mean_squared_error(label.reshape(1, -1), prediction.reshape(1, -1))
