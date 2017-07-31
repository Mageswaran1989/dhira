import logging
from itertools import islice
import random
import numpy as np

logger = logging.getLogger(__name__)

class DataManager():
    """
    This class is to act as a centralized place
    to do high-level operations on your data (i.e. loading them from filename
    to NumPy arrays).
    """

    def __init__(self, dataset, pickle_directory='dhira_pickle_folder'):
        # self.data_indexer = DataIndexer()
        self.dataset_type = dataset
        self.data_indexer_fitted = False
        self.training_data_max_lengths = {}
        self.pickle_directory = pickle_directory

    @staticmethod
    def get_random_feature(get_feature_generator, total_num_features):
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

    def get_train_data_from_file(self, max_features=None):

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
        :param filenames: : List[str]
            A collection of filenames to read the specific self.feature_type
            from, line by line.
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
        self.dataset_type.load_train_features_from_file()
        if max_features:
            logger.info("Truncating the training dataset "
                        "to {} features".format(max_features))
            self.dataset_type.train_features = self.dataset_type.truncate(self.dataset_type.train_features ,
                                                                          max_features)

        training_dataset_size = len(self.dataset_type.train_features)

        return self.dataset_type.get_train_batch_generator, training_dataset_size

    #--------------------------------------------------------------------------------------

    def get_validation_data_from_file(self, max_features=None):
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
        :param filenames: List[str]
            A collection of filenames to read the specific self.feature_type
            from, line by line.
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

        self.dataset_type.load_val_features_from_file()
        if max_features:
            logger.info("Truncating the validation dataset "
                        "to {} features".format(max_features))
            self.dataset_type.val_features = self.dataset_type.truncate(self.dataset_type.val_features, max_features)

        validation_dataset_size = len(self.dataset_type.val_features)

        return self.dataset_type.get_validation_batch_generator, validation_dataset_size

    # --------------------------------------------------------------------------------------

    def get_test_data_from_file(self, max_features=None):
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
        :param filenames: List[str]
            A collection of filenames to read the specific self.feature_type
            from, line by line.
        :param max_features: int, default=None
            If not None, the maximum number of features to produce as
            training data. If necessary, we will truncate the dataset.
            Useful for debugging and making sure things work with small
            amounts of data.
        :param max_lengths: dict from str to int, default=None
            If not None, the max length of a sequence in a given dimension.
            The keys for this dict must be in the same format as
            the features' get_lengths() function. These are the lengths
            that the features are padded or truncated to.
        :param pad: boolean, default=True
            If True, pads or truncates the features to either the input
            max_lengths or max_lengths used on the train filenames. If False,
            no padding or truncation is applied.
        :param mode: str, optional (default="word")
            String describing whether to return the word-level representations,
            character-level representations, or both. One of "word",
            "character", or "word+character"
        :return: output: returns a function to construct a test data generator
            This returns a function that can be called to produce a tuple of
            (feature generator, test_set_size). The feature generator
            outputs features as generated by the as_testing_data function
            of the underlying feature class. The test_set_size is the number
            of features in the test set, which can be used to initialize a
            progress bar.
        """

        self.dataset_type.load_test_features_from_file()
        if max_features:
            logger.info("Truncating the training dataset "
                        "to {} features".format(max_features))
            self.dataset_type.test_features = self.dataset_type.truncate(self.dataset_type.test_features ,
                                                                          max_features)

        test_dataset_size = len(self.dataset_type.test_features)

        return self.dataset_type.get_test_batch_generator, test_dataset_size

