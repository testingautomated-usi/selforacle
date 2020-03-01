import logging
import pickle

import keras
import math
import numpy
from sklearn.externals import joblib

import utils_logging
from detectors.single_image_based_detectors.abs_single_image_autoencoder import AbstractSingleImageAD
from detectors.single_image_based_detectors.deeproad_rebuild.principal_component_analysis import \
    PrincipalComponentAnalysis
from detectors.single_image_based_detectors.deeproad_rebuild.vgg_feature_extractor import VggFeatureExtractor

logger = logging.Logger("DeeproadAD")
utils_logging.log_info(logger)


class Deeproad(AbstractSingleImageAD):

    def get_input_shape(self):
        logger.error("get_input_shape is not applicable for Deeproad Models")
        exit(1)

    def __init__(self, name: str, args):
        logger.setLevel('INFO')
        self.vgg_feature_extractor = VggFeatureExtractor(args=args)
        self.pca = None  # Initialized in the train method
        super(Deeproad, self).__init__(name=name, is_keras_model=False, args=args)

    def initialize(self):
        logger.info("No specific initialization for Deeproad AD required, we're doing all of that in the train method")

    def _save_trained_model(self, path_on_disk):
        joblib.dump(self.pca, path_on_disk)

    def _load_existing_model(self, path_on_disk):
        self.pca = joblib.load(path_on_disk)
        # Note: if loading of large pca object fails, here's the workaround:
        # https://github.com/joblib/joblib/pull/920/files

    def _train_model(self, x_train, y_train, x_validation, data_dir, args):
        logger.info("Started reading images and calculating VGG based feature extraction")
        vgg_based_values = self.__vgg_extraction(x_train, data_dir=data_dir)

        logger.info("Calculated VVG Feature Arrays. Start with PCA now. This may take long and requires a lot of memory!")
        self.__train_pca(vgg_based_values, x_train)

    def __train_pca(self, vgg_based_values, x_train):
        # According to the experiment in Zhang (2018) with Deeproad_{IV}, we set the number of dimensions to 3
        # Note that Zhang (2018) is somewhat ambiguous on that as they are also mentioning Deeproad_{IV} with
        # dimension 2 - a setting which appears they have then not used in their experiments
        number_of_dimensions = 3
        number_of_neighbors = self._calc_number_of_neighbors(x_train)
        # Note: This automatically starts the training and thus may take long
        self.pca = PrincipalComponentAnalysis(number_of_dimensions=number_of_dimensions,
                                              number_of_closest_values=number_of_neighbors,
                                              data=vgg_based_values)

    def __vgg_extraction(self, x_train, data_dir: str):
        result = []
        i = 0
        for img_path in x_train:
            i = self.__logging_progress(i, x_train)
            extracted_array = self.vgg_feature_extractor.compute_feature_vector(path_to_image=img_path,
                                                                                data_dir=data_dir)
            result.append(extracted_array)
        return numpy.asarray(result)

    def __logging_progress(self, i, x_train):
        if i % 200 == 0:
            logger.info("vgg extract for image " + str(i) + " of " + str(len(x_train)))
        return i + 1

    def normalize_and_reshape(self, x) -> numpy.array:
        logger.error("get_batch_generator is not applicable for Deeproad Models")
        exit(1)

    def _calc_number_of_neighbors(self, feature_vectors: numpy.array):
        #   According to the experiment in Zhang (2018), we set the number of considered closest neighbours for distance
        #   calculation to 1/6th of the dataset
        return math.floor(len(feature_vectors) * (1 / 6))

    def _create_keras_model(self, args=None):
        logger.error("get_batch_generator is not applicable for Deeproad Models")
        exit(1)

    def get_batch_generator(self, x, y, args):
        logger.error("get_batch_generator is not applicable for Deeproad Models")
        exit(1)

    def _compile(self):
        logger.error("_compile is not applicable for Deeproad Models")
        exit(1)

    def _calc_losses_for_batch(self, x: numpy.array, labels: numpy.array, batch_size: int) -> numpy.array:
        logger.error("_compile is not applicable for Deeproad Models")
        exit(1)

    def calc_losses(self, inputs: numpy.array, labels: numpy.array, data_dir: str) -> numpy.array:
        assert inputs.shape == (len(inputs),)
        # labels are not applicable in the deeproad approach
        results = []
        count = 0
        for img_path in inputs:
            feature_vector = self.vgg_feature_extractor.compute_feature_vector(path_to_image=img_path, data_dir=data_dir)
            feature_vector = feature_vector.reshape(1, -1) # must be two dimensional: 1 sample, many features
            distance = self.pca.calculate_distance(feature_vector)
            results.append(distance)
            count = self.__logging_progress(count, inputs)
        return numpy.asarray(results)


