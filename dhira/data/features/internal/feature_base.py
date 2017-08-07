"""
This module contains the base ``Feature`` classes that concrete classes
inherit from.

``Feature``, that just exists as a base type with no functionality
"""

class FeatureBase:
    """
    A data feature, used either for training a neural network or for
    testing one.

    :param label : boolean or class (binary/multiclass)
        The label encodes the ground truth label of the Feature.
        This encoding varies across tasks and features. If we are
        making predictions on an unlabeled test set, the label is None.
    """
    def __init__(self, label=None):
        self.label = label

    def as_training_data(self):
        """
        Convert this ``Feature`` to NumPy arrays suitable for use as
        training data to models.

        :returns train_data : (set(inputs), set(label))
            The ``Feature`` as NumPy arrays to be used in the model.
            Note that ``inputs`` might itself be a complex tuple, depending
            on the ``Feature`` type.
        """
        raise NotImplementedError

    def as_testing_data(self):
        """
        Convert this ``Feature`` to NumPy arrays suitable for use as
        testing data for models.

        :return test_data : set(inputs)
            The ``Feature`` as NumPy arrays to be used in getting
            predictions from the model. Note that ``inputs`` might itself
            be a complex tuple, depending on the ``Instance`` type.
        """
        raise NotImplementedError

    def as_validation_data(self):
        """
        Convert this ``Feature`` to NumPy arrays suitable for use as
        validation data for models.

        :return val_data : (set(inputs), set(label))
            The ``Feature`` as NumPy arrays to be used in getting
            predictions from the model. Note that ``inputs`` might itself
            be a complex tuple, depending on the ``Instance`` type.
        """
        raise NotImplementedError