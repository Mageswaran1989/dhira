
import random
from itertools import islice

import numpy as np

from dhira.data.dataset.internal.text import TextDataset
from dhira.data.embedding_manager import EmbeddingManager
from dhira.data.features.internal.feature_base import FeatureBase
import logging
logger = logging.getLogger(__name__)

class DataManager():
    """
    This class is to act as a centralized place
    to do high-level operations on your data (i.e. loading them from filename
    to NumPy arrays).
    """

    def __init__(self, dataset, nlp=None):
        self.dataset = dataset
        self.training_data_max_lengths = {}
        self.nlp = nlp
        self.embedding_matrix = None

        if isinstance(self.dataset, TextDataset):
            if self.nlp is None:
                raise RuntimeError("Need spaCy nlp pipeline. eg: spacy.load('en_core_web_md') ")
            self.dataset.set_nlp_pipelie(self.nlp)


    @staticmethod
    def to_batch(features):
        """
        Converts custom/user created features from 'FeatureBase' derived class to batches
        :param features: List of features 
        :return: 
        """
        if isinstance(features, list) is not True:
            logger.info('converting feature "{}" to list of features'.format(features))
            features = [features]
        if isinstance(features[0], FeatureBase) is not True:
            logger.info("Found: {}".format(features[0]))
            raise TypeError('features in the list are not of base type `FeatureBase`')

        features = [feature.as_training_data() for feature in features]

        flattened = ([ins[0] for ins in features],
                     [ins[1] for ins in features])
        flattened_inputs, flattened_targets = flattened

        batch_inputs = tuple(map(np.asarray, tuple(zip(*flattened_inputs))))
        batch_targets = tuple(map(np.asarray, tuple(zip(*flattened_targets))))
        single_feature = batch_inputs, batch_targets
        return single_feature

    @staticmethod
    def get_random_feature(get_feature_generator, total_num_features):
        random.seed(None)
        random_pos = random.randint(1, total_num_features)
        feature_generator = get_feature_generator()
        batched_features = list(islice(feature_generator, random_pos, random_pos + 1))
        flattened = ([ins[0] for ins in batched_features],
                     [ins[1] for ins in batched_features])
        flattened_inputs, flattened_targets = flattened

        batch_inputs = tuple(map(np.asarray, tuple(zip(*flattened_inputs))))
        batch_targets = tuple(map(np.asarray, tuple(zip(*flattened_targets))))
        single_feature = batch_inputs, batch_targets
        return single_feature

    @staticmethod
    def get_batch_generator(get_feature_generator, batch_size):
        """
        Convenience function that, when called, produces a generator that yields
        individual features as numpy arrays into a generator
        that yields batches of features.
        :param get_feature_generator:  numpy array generator
            The feature_generator should be an infinite generator that outputs
            individual training features (as numpy arrays in this codebase,
            but any iterable works). The expected format is:
            ((input0, input1,...), (target0, target1, ...))
        :param batch_size: : int, optional
            The size of each batch. Depending on how many
            features there are in the dataset, the last batch
            may have less features.
        :return: returns a tuple of 2 tuples
            The expected return schema is:
            ((input0, input1, ...), (target0, target1, ...),
            where each of "input*" and "target*" are numpy arrays.
            The number of rows in each input and target numpy array
            should be the same as the batch size.
        """

        # batched_features is a list of batch_size features, where each
        # feature is a tuple ((inputs), targets)
        feature_generator = get_feature_generator()
        batched_features = list(islice(feature_generator, batch_size))
        while batched_features:
            # Take the batched features and create a batch from it.
            # The batch is a tuple ((inputs), targets), where (inputs)
            # can be (inputs0, inputs1, etc...). each of "inputs*" and
            # "targets" are numpy arrays.
            flattened = ([ins[0] for ins in batched_features],
                         [ins[1] for ins in batched_features])
            flattened_inputs, flattened_targets = flattened

            batch_inputs = tuple(map(np.asarray, tuple(zip(*flattened_inputs))))
            batch_targets = tuple(map(np.asarray, tuple(zip(*flattened_targets))))

            yield batch_inputs, batch_targets
            batched_features = list(islice(feature_generator, batch_size))

    def get_train_data(self, max_features=None):

        """
        Given a filename or list of filenames, return a generator for producing
        individual features of data ready for use in a model read from those
        file(s).

        Given a string path to a file in the format accepted by the feature,
        we fit the data_indexer word dictionary on it. Next, we use this
        DataIndexer to convert the feature into IndexedInstances (replacing
        words with integer indices).

        This function returns a function to construct generators that take
        these IndexedInstances, pads them to the appropriate lengths (either the
        maximum lengths in the dataset, or lengths specified in the constructor),
        and then converts them to NumPy arrays suitable for training with
        feature.as_training_data. The generator yields one feature at a time,
        represented as tuples of (inputs, labels).
        :param max_features: int, default=None
            If not None, the maximum number of features to produce as
            training data. If necessary, we will truncate the dataset.
            Useful for debugging and making sure things work with small
            amounts of data.

        :return: output: returns a function to construct a train data generator
            This returns a function that can be called to produce a tuple of
            (feature generator, train_set_size). The feature generator
            outputs features as generated by the as_training_data function
            of the underlying feature class. The train_set_size is the number
            of features in the train set, which can be used to initialize a
            progress bar.
        """
        # Call the overloaded function only when the provided data set doesn't load the features by default,
        # this can happen if the data set decided to postpone the loading by reading the corresponding files
        if self.dataset.train_features is None:
            self.dataset.load_train_features()

        if isinstance(self.dataset, TextDataset):
            self.dataset.fit_data_indexer()
            self.dataset.index_train_features()
            self.embedding_matrix = EmbeddingManager.get_spacy_embedding_matrix(self.nlp, self.dataset.data_indexer)

        self.dataset.pickle_train_features()

        if max_features:
            logger.info("Truncating the training dataset "
                        "to {} features".format(max_features))
            self.dataset.train_features = self.dataset.truncate(self.dataset.train_features ,
                                                                          max_features)
        training_dataset_size = len(self.dataset.train_features)

        return self.dataset.get_train_batch_generator, training_dataset_size

    #--------------------------------------------------------------------------------------

    def get_validation_data(self, max_features=None):
        """
        Given a filename or list of filenames, return a generator for producing
        individual features of data ready for use as validation data in a
        model read from those file(s).

        Given a string path to a file in the format accepted by the feature,
        we use a data_indexer previously fitted on train data. Next, we use
        this DataIndexer to convert the feature into IndexedInstances
        (replacing words with integer indices).

        This function returns a function to construct generators that take
        these IndexedInstances, pads them to the appropriate lengths (either the
        maximum lengths in the dataset, or lengths specified in the constructor),
        and then converts them to NumPy arrays suitable for training with
        feature.as_validation_data. The generator yields one feature at a time,
        represented as tuples of (inputs, labels).

        :param max_features: int, default=None
            If not None, the maximum number of features to produce as
            training data. If necessary, we will truncate the dataset.
            Useful for debugging and making sure things work with small
            amounts of data.
        :return: output: returns a function to construct a validation data generator
            This returns a function that can be called to produce a tuple of
            (feature generator, validation_set_size). The feature generator
            outputs features as generated by the as_validation_data function
            of the underlying feature class. The validation_set_size is the number
            of features in the validation set, which can be used to initialize a
            progress bar.
        """

        # Call the overloaded function only when the provided data set doesn't load the features by default,
        # this can happen if the data set decided to postpone the loading by reading the corresponding files
        if self.dataset.val_features is None:
            self.dataset.load_val_features()

        if isinstance(self.dataset, TextDataset):
            self.dataset.index_val_features()

        self.dataset.pickle_val_features()

        if max_features:
            logger.info("Truncating the validation dataset "
                        "to {} features".format(max_features))
            self.dataset.val_features = self.dataset.truncate(self.dataset.val_features, max_features)

        validation_dataset_size = len(self.dataset.val_features)

        return self.dataset.get_validation_batch_generator, validation_dataset_size

    # --------------------------------------------------------------------------------------

    def get_test_data(self, max_features=None):
        """
        Given a filename or list of filenames, return a generator for producing
        individual features of data ready for use as model test data.

        Given a string path to a file in the format accepted by the feature,
        we use a data_indexer previously fitted on train data. Next, we use
        this DataIndexer to convert the feature into IndexedInstances
        (replacing words with integer indices).

        This function returns a function to construct generators that take
        these IndexedInstances, pads them to the appropriate lengths (either the
        maximum lengths in the dataset, or lengths specified in the constructor),
        and then converts them to NumPy arrays suitable for training with
        feature.as_testinging_data. The generator yields one feature at a time,
        represented as tuples of (inputs, labels).

        :param max_features: int, default=None
            If not None, the maximum number of features to produce as
            training data. If necessary, we will truncate the dataset.
            Useful for debugging and making sure things work with small
            amounts of data.

        :return: output: returns a function to construct a test data generator
            This returns a function that can be called to produce a tuple of
            (feature generator, test_set_size). The feature generator
            outputs features as generated by the as_testing_data function
            of the underlying feature class. The test_set_size is the number
            of features in the test set, which can be used to initialize a
            progress bar.
        """

        # Call the overloaded function only when the provided data set doesn't load the features by default,
        # this can happen if the data set decided to postpone the loading by reading the corresponding files
        if self.dataset.test_features is None:
            self.dataset.load_test_features()

        if isinstance(self.dataset, TextDataset):
            self.dataset.index_test_features()

        self.dataset.pickle_test_features()

        if max_features:
            logger.info("Truncating the training dataset "
                        "to {} features".format(max_features))
            self.dataset.test_features = self.dataset.truncate(self.dataset.test_features ,
                                                                          max_features)

        test_dataset_size = len(self.dataset.test_features)

        return self.dataset.get_test_batch_generator, test_dataset_size

