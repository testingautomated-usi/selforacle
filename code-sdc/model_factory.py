import logging

import keras

from models import chauffeur, epoch, nvidia_dave2
from models.chauffeur import Chauffeur
from models.epoch import Epoch
from models.nvidia_dave2 import Dave2

logger = logging.Logger("model_factory")


def get_model(args) -> keras.Model:
    model_name = args.model
    if model_name == chauffeur.NAME:
        model_provider = Chauffeur()
    elif model_name == epoch.NAME:
        model_provider = Epoch()
    elif model_name == nvidia_dave2.NAME:
        model_provider = Dave2()
    else:
        logger.error("Model not found")
        exit(1)
        model_provider = None  # Unconfuse Jetbrains
    return model_provider.get_model(args)
