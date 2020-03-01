import json
import logging
import os
from pathlib import Path

import numpy
from scipy.stats import gamma

import utils_logging

THRESHOLDS_LOCATION = "/../models/trained-anomaly-detectors/thresholds/thresholds"

logger = logging.Logger("utils_thresholds")
utils_logging.log_info(logger)


def load_thresholds(model_class: str) -> dict:
    """
    Loads the previously stored threshold from the file system.
    This method is called if the AD model was restored from the file system.
    :param model_class: the identifier of the anomaly detector type
    :return: a dictionary of where key = threshold_identifier and value = threshold_value
    """
    json_filename = str(Path.cwd()) + THRESHOLDS_LOCATION + model_class + ".json"

    if os.path.exists(json_filename) == False:
        raise ValueError('Threshold file ' + json_filename + ' not found.')

    with open(json_filename, 'r') as fp:
        thresholds = json.loads(fp.read())

    logger.info("Loaded thresholds in " + json_filename)

    return thresholds


def calc_and_store_thresholds(losses: numpy.array, model_class: str) -> dict:
    """
    Calculates all thresholds stores them on a file system
    :param losses: array of shape (n,),
                    where n is the number of training data points, containing the losses calculated for these points
    :param model_class: the identifier of the anomaly detector type
    :return: a dictionary of where key = threshold_identifier and value = threshold_value
    """

    logger.info("Fitting reconstruction error distribution of %s using Gamma distribution params" % model_class)

    shape, loc, scale = gamma.fit(losses, floc=0)

    thresholds = {}

    conf_intervals = [0.68, 0.90, 0.95, 0.99, 0.999, 0.9999, 0.99999]

    logger.info("Creating thresholds using the confidence intervals: %s" % conf_intervals)

    for c in conf_intervals:
        thresholds[str(c)] = gamma.ppf(c, shape, loc=loc, scale=scale)

    as_json = json.dumps(thresholds)

    json_filename = str(Path.cwd()) + THRESHOLDS_LOCATION + model_class + ".json"

    print("Saving thresholds to %s" % json_filename)

    if os.path.exists(json_filename):
        os.remove(json_filename)

    with open(json_filename, 'a') as fp:
        fp.write(as_json)

    return thresholds
