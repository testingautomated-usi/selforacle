import logging
import os

import numpy

import utils
import utils_args
from detectors.anomaly_detector import AnomalyDetector
from detectors.img_sequence_cnnlstm.cnn_lstm_img import CnnLstmImg
from detectors.single_image_based_detectors.autoencoders.convolutional_autoencoder import ConvolutionalAutoencoder
from detectors.single_image_based_detectors.autoencoders.deep_autoencoder import DeepAutoencoder
from detectors.single_image_based_detectors.autoencoders.simple_autoencoder import SimpleAutoencoder
from detectors.single_image_based_detectors.autoencoders.variational_autoencoder import VariationalAutoencoder
from detectors.single_image_based_detectors.deeproad_rebuild.deeproad import Deeproad

logger = logging.Logger("main")


def get_model(args, model_name, data_dir):
    dataset_name = dataset_name_from_dir(data_dir)
    if model_name == 'CAE':
        return ConvolutionalAutoencoder(name="convolutional-autoencoder-model-" + dataset_name, args=args)
    elif model_name == 'SAE':
        return SimpleAutoencoder(name="simple-autoencoder-model-" + dataset_name, args=args)
    elif model_name == "VAE":
        return VariationalAutoencoder(name="variational-autoencoder-model-" + dataset_name, args=args)
    elif model_name == "DAE":
        return DeepAutoencoder(name="deep-autoencoder-model-" + dataset_name,
                               args=args, hidden_layer_dim=256)
    elif model_name == "IMG-LSTM":
        return CnnLstmImg(name="LSTM-model-" + dataset_name, args=args)

    elif model_name == "DEEPROAD":
        return Deeproad(name="deeproad-pca-model-" + dataset_name, args=args)

    else:
        logger.error("Unknown Model Type: " + model_name)


def dataset_name_from_dir(data_dir):
    dataset_name = os.path.basename(os.path.normpath(data_dir))
    return dataset_name


def main():
    args = utils_args.specify_args()
    if args.gray_scale:
        # TODO Remove or complete gray scale implementation
        logger.error("Gray Scale is not yet ready to be used")
        exit(1)
        utils.IMAGE_CHANNELS = 1

    numpy.random.seed(args.random_state)

    utils_args.store_and_print_params(args)

    for data_dir in args.data_dir:
        for model_name in args.model_name:
            print("\n --- MODEL NAME: " + model_name + " for dataset " + data_dir + " --- \n")
            # Load the correct anomaly detector class
            load_or_train_model(args, data_dir, model_name)
            print("\n --- COMPLETED  " + model_name + " for dataset " + data_dir + " --- \n")

    print("\ndone")
    # evaluate_optically_img_loss(trained=autoencoder, x_test=X_test, y_test=y_test, args=args)


def load_or_train_model(args, data_dir, model_name) -> AnomalyDetector:
    anomaly_detector = get_model(args=args, model_name=model_name, data_dir=data_dir)
    # Create and compile the ADs model
    anomaly_detector.initialize()
    # Load the image paths in a form suitable for the AD (sequence or single-image)
    #       Indicate whether this models requires special TD size
    restrict_size = args.train_abs_size_models.count(model_name) > 0
    # TODO This is inefficient. Training data should only be loaded if model not yet trained. Move this to train_model mthod
    x_train, y_train = anomaly_detector.load_img_paths(restrict_size=restrict_size, data_dir=data_dir, eval_data_mode=False)

    # Load previously trained model or train it now
    anomaly_detector.load_or_train_model(x_train=x_train, y_train=y_train, data_dir=data_dir)

    # Sanity check for loss calculator
    # anomaly_detector.calc_losses(x_train[:200], y_train[:200], data_dir=data_dir)
    return anomaly_detector

if __name__ == '__main__':
    main()
