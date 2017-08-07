import numpy as np

from dhira.data.features.internal.feature_base import FeatureBase


class ImageFeature(FeatureBase):
    """
    Class that encapsulates an image feature
    """
    def __init__(self, image, label):
        super(ImageFeature, self).__init__(label)
        self.image: np.ndarray = image

    def normalize(self, x):
        """
        Normalize a list of sample image data in the range of 0 to 1
        : x: List of image data.  The image shape is (32, 32, 3)
        : return: Numpy array of normalize data
        """
        minV = np.min(x)
        maxV = np.max(x)
        ret = (x - minV) / maxV
        return ret

    def as_training_data(self):
        return ((self.image,), (self.label,))

    def as_validation_data(self):
        return ((self.image,), (self.label,))

    def as_testing_data(self):
        return ((self.image,), (self.label,))