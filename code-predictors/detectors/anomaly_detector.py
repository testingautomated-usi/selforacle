import abc
import logging
import os

import keras
import numpy
from tensorflow.python.keras.utils import Sequence

import utils_logging
import utils_thresholds

logger = logging.Logger("AbstractAnomalyDetector")
utils_logging.log_info(logger)

class AnomalyDetector(abc.ABC):
    def __init__(self, name: str, is_keras_model: bool, args):
        self.is_keras_model = is_keras_model
        self.args = args
        self.name = name
        self.thresholds = None
        self.keras_model: keras.Model = None

    def _get_keras_model(self):
        self._assert_is_keras_type()
        if self.keras_model is None:
            logger.error("Keras Model not found. Maybe the AD is not yet initialized?")
        return self.keras_model

    @abc.abstractmethod
    def _create_keras_model(self, args=None):
        logger.error("create_autoencoder must be overriden in child class")
        exit(1)

    @abc.abstractmethod
    def normalize_and_reshape(self, x) -> numpy.array:
        logger.error("create_autoencoder must be overriden in child class")
        exit(1)

    @abc.abstractmethod
    def get_batch_generator(self, x, y, data_dir: str) -> Sequence:
        logger.error("get_batch_generator must be overriden in child class")
        exit(1)

    def _compile(self):
        self._assert_is_keras_type()
        self.keras_model.compile(optimizer='adam', loss='mean_squared_error')

    @abc.abstractmethod
    def load_img_paths(self, data_dir: str, restrict_size: bool, eval_data_mode: bool):
        logger.error("load_img_paths must be overriden in child class")
        exit(1)

    def initialize(self) -> None:
        self._assert_is_keras_type()
        self.keras_model = self._create_keras_model(args=self.args)
        self.keras_model.summary()
        self._compile()

    @abc.abstractmethod
    def get_input_shape(self):
        logger.error("load_img_paths must be overriden in child class")
        exit(1)

    def load_or_train_model(self, x_train, y_train, data_dir: str):
        model_class = self.__class__.__name__.lower()
        dataset_name = os.path.basename(os.path.normpath(data_dir))
        model_name_on_disk = "../models/trained-anomaly-detectors/" + dataset_name + "-" + model_class + ".h5"
        if os.path.exists(model_name_on_disk) and self.args.delete_trained:
            os.remove(model_name_on_disk)
        if os.path.exists(model_name_on_disk):
            print("loading existing model")
            try:
                self._load_existing_model(model_name_on_disk)
            except:
                self._treat_model_loading_problem(data_dir, model_class, model_name_on_disk, x_train, y_train)
            if self.args.always_calc_thresh:
                self._calc_and_store_thresholds(x_train=x_train, y_train=y_train, data_dir=data_dir,
                                                model_class=model_class)
            else:
                self._load_thresholds(data_dir, model_class)
        else:
            self._train_model(x_train=x_train,
                              y_train=y_train,
                              x_validation=None,
                              data_dir=data_dir,
                              args=self.args)
            self._try_saving(model_name_on_disk)
            self._calc_and_store_thresholds(x_train, y_train, data_dir, model_class)

    def _treat_model_loading_problem(self, data_dir, model_class, model_name_on_disk, x_train, y_train):
        logger.warning(
            "Could not load model (most likely its invalid) " + model_name_on_disk + "Will train without saving now")
        self._train_model(x_train=x_train,
                          y_train=y_train,
                          x_validation=None,
                          data_dir=data_dir,
                          args=self.args)
        os.remove(model_name_on_disk)
        self._try_saving(model_name_on_disk)
        self._calc_and_store_thresholds(x_train, y_train, data_dir, model_class)

    def _try_saving(self, model_name_on_disk):
        try:
            self._save_trained_model(model_name_on_disk)
        except:
            logger.warning("Could not save model (most likely as it is too big). " + model_name_on_disk)

    def _save_trained_model(self, path_on_disk):
        self._assert_is_keras_type()
        self.keras_model.save(path_on_disk)

    def _load_existing_model(self, path_on_disk):
        self._assert_is_keras_type()
        self.keras_model.load_weights(path_on_disk)

    def _assert_is_keras_type(self):
        # keras is default. if subclass uses something else, this method has to be overwritten
        assert self.is_keras_model

    def _train_model(self, x_train, y_train, x_validation, data_dir: str, args):
        self._assert_is_keras_type()

        train_generator = self.get_batch_generator(x=x_train, y=y_train, data_dir=data_dir)

        self.keras_model.fit_generator(generator=train_generator,
                                       epochs=args.nb_epoch,
                                       use_multiprocessing=False, )

        self._post_training(self.keras_model, x_train, y_train, data_dir, args)

    def calc_losses(self, inputs: numpy.array, labels: numpy.array, data_dir: str) -> numpy.array:
        """
        Calculates losses for all input and labels passed. The data in the passed arrays are img paths
        :param inputs: Input paths (one dimensional for single images, two dimensional for series)
        :param labels: Paths to images to validate against. Ignored for autoencoders.
        :return: List of losses, in same order as inputs
        """
        # Note: This would fail for deeproad. Hence, deeproad will override this method
        batch_generator = self.get_batch_generator(x=inputs, y=labels, data_dir=data_dir)

        prediction_batch_size = batch_generator.get_batch_size()
        result = None
        for index in range(batch_generator.__len__()):
            x, y = batch_generator.__getitem__(index=index)
            losses = self._calc_losses_for_batch(x, y, prediction_batch_size)
            if result is None:
                result = losses
            else:
                result = numpy.concatenate((result, losses))

            if index % 10 == 0:
                logger.info("predicting batch "+ str(index) + " out of " + str(batch_generator.__len__()))

        return result

    def _calc_losses_for_batch(self, x: numpy.array, labels: numpy.array, batch_size: int) -> numpy.array:
        self._assert_is_keras_type()
        # Apply Keras Prediction
        predictions = self.keras_model.predict(x=x, batch_size=batch_size)
        assert labels.shape == predictions.shape
        # Calculate distances
        distances = numpy.empty(shape=(len(labels),))
        for i in range(len(labels)):
            distance = self._distance_metrics(labels[i], predictions[i])
            distances.put(i, distance)
        return distances

    def _distance_metrics(self, label: numpy.array, prediction: numpy.array):
        # Use euclid as default for evaluation
        return numpy.sqrt(numpy.sum((label - prediction) ** 2))

    def _load_thresholds(self, data_dir: str, model_class: str):
        return utils_thresholds.load_thresholds(model_class=model_class)

    def _calc_and_store_thresholds(self, x_train: numpy.array, y_train: numpy.array, data_dir: str, model_class: str):
        logger.info("Calculating losses...")
        losses = self.calc_losses(inputs=x_train, labels=y_train, data_dir=data_dir)
        logger.info("...Done. Calculating thresholds now...")
        return utils_thresholds.calc_and_store_thresholds(losses=losses, model_class=model_class)

    # Used for debugging
    def _post_training(self, keras_model, x, y, data_dir, args):
        pass
