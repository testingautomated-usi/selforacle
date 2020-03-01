import logging

import cv2
import keras
import numpy
from keras import Model
from keras.layers import Flatten

import utils
import utils_logging

logger = logging.Logger("VggFeatureExtractor")
utils_logging.log_info(logger)

OUTPUT_PRECISION = numpy.float32


class VggFeatureExtractor:

    def __init__(self, args):
        self.args = args

        if OUTPUT_PRECISION != numpy.float64:
            logger.warning("Results of VGG Feature extractor are casted from float64 to " + str(OUTPUT_PRECISION))

        # VGG only works with exactly 3 channels, see
        # https://keras.io/applications/#vgg16
        assert utils.IMAGE_CHANNELS == 3

        vgg19 = keras.applications.vgg19.VGG19()
        self.partial_model_3_2 = self.get_partial_model(layer_to_extract_activations="block3_conv2", vgg=vgg19)
        self.partial_model_4_1 = self.get_partial_model(layer_to_extract_activations="block4_conv1", vgg=vgg19)

    def get_partial_model(self, layer_to_extract_activations: str, vgg: keras.Model) -> keras.Model:
        output_layer = vgg.get_layer(layer_to_extract_activations).output
        f0 = Flatten()(output_layer)
        return Model(inputs=vgg.input, outputs=f0)

    def compute_feature_vector(self, path_to_image: str, data_dir: str) -> numpy.array:
        """
        Computes the feature vector, as described in Zhang(2018), section 3.2.2
        :param path_to_image: Path to the image on the local file system for which the feature vector should be caculated
        :return: A one dimensional feature vector
        """

        img = self._load_image_from_path(image_name=path_to_image, data_dir=data_dir)

        # Alternatively, we could change VGG to take another input size
        # Zhang (2018) don't specify what they do - we picked this way, allowing us to leave vgg16 at its defaults
        img = self._resize_image_for_vgg_16(img)

        # "DeepRoadIV inputs a snowy image to VGGNet,
        # and chooses the convolutional layer conv 4_2 and conv 5_3
        # to extract content and style features respectively."  (Zhang, 2018)
        activations_3_2 = self._activation_at_conv_layer_3_2(img=img)
        activations_4_1 = self._activation_at_conv_layer_4_1(img=img)

        content_features_3_2 = self._extract_content_features(activations=activations_3_2)
        content_features_4_1 = self._extract_content_features(activations=activations_4_1)

        # "the style feature G 5_3 is computed by Equation 7"  (Zhang, 2018)
        style_features_4_1 = self._compute_style_vector(features=content_features_4_1)

        # "Then, matrix F 4_2 and G 5_3 are flattened and concatenated to feature vector v" (Zhang, 2018)
        feature_vector = self._flatten_and_concatenate_activations(activations_4_3=content_features_3_2,
                                                                   style_array_5_3=style_features_4_1)

        return feature_vector

    def _extract_content_features(self, activations) -> numpy.array:
        assert len(activations.shape) == 3

        # Utility Variables
        number_of_features = activations.shape[0]
        feature_map_width_wi = activations.shape[1]
        feature_map_height_hi = activations.shape[2]
        feature_length = feature_map_width_wi * feature_map_height_hi
        # VGG19 convolutional layers have one of the following number of filters (=features)
        assert number_of_features == 512 or number_of_features == 256 or number_of_features == 128 or number_of_features == 64

        # Extracting features
        result = numpy.empty((number_of_features, feature_length))
        for feature_count in range(number_of_features):  # 'depth' of convolutional layer
            feature = activations[feature_count, ::, ::]
            flat_feature = feature.flatten()
            result.put(feature_count, flat_feature)
        return result

    def _flatten_and_concatenate_activations(self, activations_4_3: numpy.array, style_array_5_3: numpy.array):
        flat_4_3 = activations_4_3.flatten()
        flat_5_3 = style_array_5_3.flatten()
        # Set precision of target array from float64 to float32 to save some memory (this array is stored for a long time)
        concatenated = numpy.concatenate((flat_4_3, flat_5_3))
        result_32 = concatenated.astype(dtype=numpy.float32, casting='same_kind')
        return result_32

    def _compute_style_vector(self, features: numpy.array):
        """
        "...style information aims at capturing the texture of images and it is defined by feature correlation, which
        can be computed by the Gram matrix..." (Zhang, 2018)
        :param features: a one dimensional array of activations to calculate the style vector for
        :return: the gram matrix of the input (dot product of the input and the transposed input)
        """
        # https://stackoverflow.com/questions/50733148/numpy-efficient-matrix-self-multiplication-gram-matrix
        return features.dot(features.T)

    def _activation_at_conv_layer_3_2(self, img: numpy.array):
        activations = self.partial_model_3_2.predict(img)
        onedim = activations[0]
        three_dim = onedim.reshape(256, 56, 56, )
        assert activations.size == three_dim.size
        return three_dim

    def _activation_at_conv_layer_4_1(self, img: numpy.array):
        activations = self.partial_model_4_1.predict(img)
        onedim = activations[0]
        three_dim = onedim.reshape(512, 28, 28, )
        assert activations.size == three_dim.size
        return three_dim

    def _load_image_from_path(self, image_name: str, data_dir: str) -> numpy.array:
        return utils.load_img_from_path(data_dir=data_dir,
                                                               image_name=image_name,
                                                               is_gray_scale=self.args.gray_scale)

    def _resize_image_for_vgg_16(self, image_input):
        """
        Resizes the image to 224x224 which is the default input size for VGG.\n
        Note: A potantially better approach would be to add a new input layer to the VGG Model,
        which accepts a custom image size.
        However, the authors of deeproad have not specified such a layer and since we attempt to rebuild their settings
        we suboptimally resize the images to the default vgg16 input size. \n
        Also note that the pictures were probably (depending on the settings) previously (when loaded) shrinked
        to something smaller than 224x224. This is not a problem as all our ADs
        work with the same image size as base input.
        :param image_input: an normalized image with three changes and shape (-1, height, width, 3)
                where height and width are arbitrary
        :return: an image of size (224 x 224 x 3) with channels_last
        """

        # Denormalize image (note: the previous normalization and now again denormalization is inefficient)
        # but more error prone in case of changes for future runs (code sharing!). Hence, I keep it like that for now
        image_data = image_input * 255
        image_data = image_data.astype('uint8')
        image_data = image_data.reshape(utils.IMAGE_HEIGHT, utils.IMAGE_WIDTH, utils.IMAGE_CHANNELS)

        vgg_image_side_px = 224
        resized_image = cv2.resize(image_data, dsize=(vgg_image_side_px, vgg_image_side_px),
                                   interpolation=cv2.INTER_CUBIC)
        resized_image = resized_image.astype('float32') / 255.
        return resized_image.reshape(-1, vgg_image_side_px, vgg_image_side_px, 3)
