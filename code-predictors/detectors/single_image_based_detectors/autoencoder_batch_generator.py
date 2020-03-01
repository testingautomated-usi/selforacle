import numpy
from keras.layers import np
from keras.utils import Sequence

from utils import load_image, augment, preprocess, resize

APPLY_DATA_AUGMENTATION = True


class AutoencoderBatchGenerator(Sequence):
    """
    Single image based batch generator. Generated inputs == generated labels (i.e., x == y) as required by autoencoders
    """

    def __init__(self, path_to_pictures: numpy.array, anomaly_detector, data_dir: str, batch_size: int):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.path_to_pictures = path_to_pictures
        self.model = anomaly_detector

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = start_index + self.batch_size
        batch_paths = self.path_to_pictures[start_index:end_index]
        input_shape = self.model.get_input_shape()
        x_shape = (len(batch_paths),) + (input_shape)
        x = np.empty(shape=x_shape)
        for i, path in enumerate(batch_paths):

            # apply augmentation to 60% of the images, if enabled
            if APPLY_DATA_AUGMENTATION and np.random.rand() < 0.6:
                # data augmentation
                img = load_image(self.data_dir, path)
                img = augment(img)
                img = preprocess(img) # crop + resize + rgb2yuv
            else:
                img = load_image(self.data_dir, path)
                img = resize(img)

            x[i] = self.model.normalize_and_reshape(img)
        return x, x

    def __len__(self):
        return len(self.path_to_pictures) // self.batch_size

    def get_batch_size(self):
        return self.batch_size