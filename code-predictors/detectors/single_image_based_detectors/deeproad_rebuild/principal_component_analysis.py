import logging
from typing import List

import numpy
from sklearn.decomposition import PCA

logger = logging.Logger("PrincipalComponentAnalysis")


class PrincipalComponentAnalysis:

    def __init__(self, number_of_dimensions: int, data: numpy.array, number_of_closest_values: int):
        assert number_of_closest_values <= len(data)
        self.number_of_closest_values = number_of_closest_values
        self.number_of_dimensions = number_of_dimensions
        self.data = data
        self.__train()

    def __train(self) -> None:
        self.pca = PCA(n_components=self.number_of_dimensions)
        logger.info("Start fitting PCA")
        self.low_dim_train_data = self.pca.fit_transform(X=self.data)
        logger.info("Completed fitting PCA")

    def calculate_distance(self, feature_array: numpy.array) -> float:
        observed_record = self._transform(feature_array)

        distances = self._calculate_distances(observed_record)
        smallest_distances = self._filter_smallest_distances(distances)
        average = self._average(smallest_distances)
        return average

    def _calculate_distances(self, observed_record) -> List[float]:
        distances = []
        for td_record in self.low_dim_train_data:
            euclid_distance = numpy.linalg.norm(td_record - observed_record)
            distances.append(euclid_distance)
        return distances

    def _transform(self, feature_array: numpy.array) -> numpy.array:
        return self.pca.transform(feature_array)

    def _filter_smallest_distances(self, distances: List[float]):
        np_distances = numpy.asarray(distances)
        indexes = numpy.argpartition(np_distances, self.number_of_closest_values)
        values = np_distances[indexes[:self.number_of_closest_values]]
        return values

    def _average(self, smallest_distances: numpy.array) -> float:
        assert smallest_distances.shape == (self.number_of_closest_values,)
        return numpy.mean(smallest_distances)
