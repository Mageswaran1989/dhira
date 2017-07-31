from .feature import TextFeature
from overrides import overrides
import numpy as np

class GloveFeature(TextFeature):
    def __init__(self, focal_input, context_input, cooccurrence_count):
        super(GloveFeature, self).__init__(None)
        self.focal_input = np.asarray(focal_input)
        self.context_input = np.asarray(context_input)
        self.cooccurrence_count = np.asarray(cooccurrence_count)

    # @classmethod
    # def read_from_line(cls, line):
    #     cls(line)

    def as_training_data(self):
        return ((self.focal_input, self.context_input, self.cooccurrence_count), (self.label,))

    def as_validation_data(self):
        return ((self.focal_input, self.context_input, self.cooccurrence_count), (self.label,))

    def as_testing_data(self):
        return ((self.focal_input, self.context_input, self.cooccurrence_count), (self.label,))