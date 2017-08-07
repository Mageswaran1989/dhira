import numpy as np

from dhira.data.features.internal.feature_base import TextFeature


class GloveFeature(TextFeature):
    """
    Special "Feature" class that holds the pre-computed word probabilities
    """
    def __init__(self, focal_input, context_input, cooccurrence_count):
        super(GloveFeature, self).__init__(None)
        #Index of the focal(word of interest) word
        self.focal_input = np.asarray(focal_input)
        #Index of word(s) around focal word
        self.context_input = np.asarray(context_input)
        #Count occurrences
        self.cooccurrence_count = np.asarray(cooccurrence_count)

    def as_training_data(self):
        return ((self.focal_input, self.context_input, self.cooccurrence_count), (self.label,))

    def as_validation_data(self):
        return ((self.focal_input, self.context_input, self.cooccurrence_count), (self.label,))

    def as_testing_data(self):
        return ((self.focal_input, self.context_input, self.cooccurrence_count), (self.label,))