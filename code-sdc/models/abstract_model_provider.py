import abc
import logging

import keras

logger = logging.Logger("AbstractModelProvider")


class AbstractModelProvider(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_model(self, args) -> keras.Model:
        logger.error("Method must be overriden in child class")
        exit(1)

    @abc.abstractmethod
    def get_name(self) -> str:
        logger.error("Method must be overriden in child class")
        exit(1)

