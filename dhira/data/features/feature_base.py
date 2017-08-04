"""
This module contains the base ``Feature`` classes that concrete classes
inherit from.

Specifically, there are three classes:
1. ``Feature``, that just exists as a base type with no functionality

2. ``TextFeature``, which adds a ``words()`` method and a method to convert
   strings to indices using a DataIndexer.

3. ``IndexedFeature``, which is a ``TextFeature`` that has had all of its
   strings converted into indices.

This class has methods to deal with padding (so that sequences all have the
same length) and converting an ``Feature`` into a set of Numpy arrays suitable
for use with TensorFlow.
"""

class FeatureBase:
    """
    A data feature, used either for training a neural network or for
    testing one.

    :param label : boolean or index
        The label encodes the ground truth label of the Feature.
        This encoding varies across tasks and features. If we are
        making predictions on an unlabeled test set, the label is None.
    """
    def __init__(self, label=None):
        self.label = label

    def as_training_data(self):
        """
        Convert this ``IndexedFeature`` to NumPy arrays suitable for use as
        training data to models.

        :returns train_data : (inputs, label)
            The ``IndexedFeature`` as NumPy arrays to be used in the model.
            Note that ``inputs`` might itself be a complex tuple, depending
            on the ``Instance`` type.
        """
        raise NotImplementedError

    def as_testing_data(self):
        """
        Convert this ``IndexedInstance`` to NumPy arrays suitable for use as
        testing data for models.

        :return test_data : inputs
            The ``IndexedInstance`` as NumPy arrays to be used in getting
            predictions from the model. Note that ``inputs`` might itself
            be a complex tuple, depending on the ``Instance`` type.
        """
        raise NotImplementedError

    def as_validation_data(self):
        raise NotImplementedError