import logging

import numpy
from keras.layers import np
from keras.utils import Sequence

import utils
import utils_logging
from utils import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS

FRAME_INTERVAL = 15

logger = logging.Logger("LstmImgBatchGenerator")
utils_logging.log_info(logger)

class LstmImgBatchGenerator(Sequence):

    def __init__(self, x_paths_to_pictures, y_paths_to_pictures, data_dir: str, sequence_length: int, gray_scale: bool):
        self.data_dir = data_dir
        logger.warning("Using hard-coded batch size in lstm img batch generator")
        self.batch_size = 4
        self.x_paths_to_pictures = x_paths_to_pictures
        self.y_paths_to_pictures = y_paths_to_pictures
        self.sequence_length = sequence_length
        self.gray_scale= gray_scale

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = start_index + self.batch_size
        this_batch_x_paths = self.x_paths_to_pictures[start_index:end_index]
        this_batch_y_paths = self.y_paths_to_pictures[start_index:end_index]

        assert this_batch_x_paths.size == this_batch_y_paths.size
        assert this_batch_x_paths[0][1] == this_batch_y_paths[0][0]

        x_images = self.empty_batch_array(this_batch_x_paths)
        for i, paths in enumerate(this_batch_x_paths):
            x = self.load_paths_to_images(paths=paths)
            x_images[i] = x
        y_images = self.empty_batch_array(this_batch_y_paths)
        for i, paths in enumerate(this_batch_y_paths):
            x = x_images[i]
            x_sublist = x[1:]
            last_y = paths[len(paths)-1]
            last_img = self.load_paths_to_images(numpy.asarray([last_y]))
            y = numpy.concatenate((x_sublist, last_img))
            assert len(x) == len(y)
            y_images[i] = y

        assert numpy.array_equal(x_images[0][1], y_images[0][0])

        return x_images, y_images

    def empty_batch_array(self, this_batch_y_paths):
        return np.empty([len(this_batch_y_paths), self.sequence_length, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])

    def load_paths_to_images(self, paths):
        images = np.empty([len(paths), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])

        for k, path in enumerate(paths):
            images[k] = utils.load_img_from_path(data_dir=self.data_dir,
                                                                        image_name=path,
                                                                        is_gray_scale=self.gray_scale)
        return images


    def get_single_sequence(self, index):
        x = self.x_paths_to_pictures[index]
        x_imgs = []
        x_imgs.append(self.load_paths_to_images(x))
        x_imgs = numpy.array(x_imgs)
        # Low performance implementation, but this is not relevant as we're only using the method for manual check
        y = self.y_paths_to_pictures[index]
        y_imgs = self.load_paths_to_images(y)
        assert len(y_imgs) == self.sequence_length
        next_image = y_imgs[self.sequence_length - 1]
        return x_imgs, next_image

    def __len__(self):
        return (len(self.x_paths_to_pictures ) - FRAME_INTERVAL)// self.batch_size

    def get_batch_size(self):
        return self.batch_size


