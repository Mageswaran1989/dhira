from .feature import Feature

class ImageFeature(Feature):
    """
    Class that encapsulates the common functionality for an image features
    """
    def __init__(self, image, label):
        super(Feature, self).__init__(label)
        self.image = image

# class BatchImageFeatures