import logging
import os
from abc import ABC

import numpy
import pandas as pd

import utils_logging
from detectors.anomaly_detector import AnomalyDetector
from detectors.single_image_based_detectors.autoencoder_batch_generator import AutoencoderBatchGenerator

logger = logging.Logger("SingleImageAD")
utils_logging.log_info(logger)


class AbstractSingleImageAD(AnomalyDetector, ABC):

    def get_batch_generator(self, x, y, data_dir: str):
        return AutoencoderBatchGenerator(path_to_pictures=x, anomaly_detector=self, data_dir=data_dir,
                                         batch_size=self.args.batch_size)

    def load_img_paths(self, data_dir: str, restrict_size: bool, eval_data_mode: bool):


        x_center = None
        x_left = None
        x_right = None
        y = None
        if eval_data_mode:
            data_df = pd.read_csv(os.path.join(data_dir, 'driving_log.csv'))
            x_center = data_df['center'].values
            frame_ids = data_df['FrameId'].values
            are_crashes = data_df['Crashed'].values

        else:
            tracks = ["track1", "track2", "track3"]
            drive = ["normal", "reverse"]
            for track in tracks:
                for drive_style in drive:
                    data_df = pd.read_csv(os.path.join(data_dir, track, drive_style, 'driving_log.csv'))
                    if x_center is None:
                        x_center = data_df['center'].values
                        x_left = data_df['left'].values
                        x_right = data_df['right'].values
                        y = data_df['steering'].values
                    else:
                        x_center = numpy.concatenate((x_center, data_df['center'].values), axis=0)
                        y = numpy.concatenate((y, data_df['steering'].values), axis=0)

        if restrict_size and len(x_center) > self.args.train_abs_size != -1 and not eval_data_mode:
            shuffle_seed = numpy.random.randint(low=1)
            numpy.random.seed(shuffle_seed)
            numpy.random.shuffle(x_center)
            numpy.random.seed(shuffle_seed)
            numpy.random.shuffle(x_left)
            numpy.random.seed(shuffle_seed)
            numpy.random.shuffle(x_right)
            numpy.random.seed(shuffle_seed)
            numpy.random.shuffle(y)
            per_image_size = int(self.args.train_abs_size / 3)
            x_center = x_center[:per_image_size]
            x_left = x_left[:per_image_size]
            x_right = x_right[:per_image_size]
            y = y[:self.args.train_abs_size]

        if eval_data_mode:
            return x_center, frame_ids, are_crashes
        else:
            print("Train dataset: " + str(len(x_center)) + " elements")
            images = numpy.concatenate((x_left, x_center, x_right))
            labels = numpy.concatenate((x_left, x_center, x_right))
            return images, labels
