"""
VGG16 feature extractor for content-based image retrieval.
Extracts normalized feature vectors from images using a pre-trained VGG16 model.
"""

import numpy as np
from numpy import linalg as LA
import logging
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from src.utils.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VGGNet:
    def __init__(self):
        """Initialize VGG16 model with pre-trained ImageNet weights."""
        config = load_config()
        self.input_shape = tuple(config['model']['input_shape'])
        self.weight = config['model']['weights']
        self.pooling = config['model']['pooling']
        try:
            self.model = VGG16(weights=self.weight, 
                                input_shape=self.input_shape, 
                                pooling=self.pooling, 
                                include_top=False)
            self.model.predict(np.zeros((1, *self.input_shape)))
            logger.info("VGG16 model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize VGG16 model: {e}")
            raise

    def extract_feat(self, img_path):
        """
        Extract normalized feature vector from an image.

        Args:
            img_path (str): Path to the input image.

        Returns:
            numpy.ndarray: Normalized feature vector of length 512.
        """
        try:
            img = image.load_img(img_path, target_size=self.input_shape[:2])
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            feat = self.model.predict(img, verbose=0)
            norm_feat = feat[0] / LA.norm(feat[0])
            logger.debug(f"Extracted features from {img_path}")
            return norm_feat
        except Exception as e:
            logger.error(f"Error extracting features from {img_path}: {e}")
            raise