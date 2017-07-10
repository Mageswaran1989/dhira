import logging
import numpy as np
import os
import errno
import pickle
from itertools import islice

from .data_indexer import DataIndexer
from .dataset import TextDataset, IndexedDataset

logger = logging.getLogger(__name__)


class DataManager():
    """
    This class is to act as a centralized place
    to do high-level operations on your data (i.e. loading them from filename
    to NumPy arrays).
    """

    def __init__(self, feature_type):
        self.data_indexer = DataIndexer()
        self.feature_type = feature_type
        self.data_indexer_fitted = False
        self.training_data_max_lengths = {}
        self.pickle_directory = 'dhira_pickle_folder'

    def set_pickle_folder(self, directory='dhira_pickle_dir'):
        """
        Set a folder to pickle(Serialize) python objects for faster reloading 
        :param folder_name: Folder name, preferably the dataset name
        :return: 
        """
        self.pickle_directory = directory
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    def check_pickle_exists(self, pickle_file):
        """
        Check whether the pickle file exists before loading
        :param pickle_file Pickle file to be checked
        :return: 'True' if exists or otherwise
        """
        return os.path.exists(self.pickle_directory + '/' + pickle_file)

    def write_pickle(self, python_object, file_name):
        """
        Pickles the processed features for faster loading
        :param file_name: File Name to store the pickle objects
        :return: Path of the file created
        """
        file = self.pickle_directory + '/' + file_name
        if (os.path.exists(file)):
            input('Found existing files!!! Press Enter to Continue...')

        file_handler = open(file, 'wb')
        pickle.dump(python_object, file_handler)

    def read_pickle(self, file_name):
        """
        Reads the pickled file and returns the python object
        :param file_name: File Name to read the pickle objects
        :return: Python object
        """
        file = self.pickle_directory + '/' + file_name
        file_handler = open(file, 'rb')
        return pickle.load(file_handler)

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
            batch_inputs = tuple(map(np.array, tuple(zip(*flattened_inputs))))
            batch_targets = tuple(map(np.array, tuple(zip(*flattened_targets))))
            yield batch_inputs, batch_targets
            batched_features = list(islice(feature_generator, batch_size))

    def get_train_data_from_file(self, filenames, min_count=1,
                                 max_features=None,
                                 max_lengths=None, pad=True, mode="word"):

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
        :param min_count: int, default=1
            The minimum number of times a word must occur in order
            to be indexed.
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
            max_lengths or max_lengths across the train filenames. If False,
            no padding or truncation is applied.
        :param  mode: str, optional (default="word")
            String describing whether to return the word-level representations,
            character-level representations, or both. One of "word",
            "character", or "word+character"
        :return: output: returns a function to construct a train data generator
            This returns a function that can be called to produce a tuple of
            (feature generator, train_set_size). The feature generator
            outputs features as generated by the as_training_data function
            of the underlying feature class. The train_set_size is the number
            of features in the train set, which can be used to initialize a
            progress bar.
        """
        if self.data_indexer_fitted:
            raise ValueError("You have already called get_train_data from "
                             "this DataManager, so you cannnot do it again. "
                             "If you want to train on multiple datasets, pass "
                             "in a list of files.")
        train_file_pickle = 'train_data.p'
        data_indexer_pickle = 'data_indexer.p'

        if(not (self.check_pickle_exists(train_file_pickle) & self.check_pickle_exists(data_indexer_pickle))):
            logger.info("Processing the train data file for first time")
            logger.info("Getting training data from {}".format(filenames))

            training_dataset = TextDataset.read_from_file(filenames,
                                                          self.feature_type)
            if max_features:
                logger.info("Truncating the training dataset "
                            "to {} features".format(max_features))
                training_dataset = training_dataset.truncate(max_features)

            # Since this is data for training, we fit the data indexer
            logger.info("Fitting data indexer word "
                        "dictionary, min_count is {}.".format(min_count))
            self.data_indexer.fit_word_dictionary(training_dataset,
                                                  min_count=min_count)
            self.data_indexer_fitted = True

            # With our fitted data indexer, we convert the dataset
            # from string tokens to numeric int indices.
            logger.info("Indexing dataset")
            indexed_training_dataset = training_dataset.to_indexed_dataset(
                self.data_indexer)

            self.write_pickle(indexed_training_dataset, train_file_pickle)
            self.write_pickle(self.data_indexer, data_indexer_pickle)
        else:
            logger.info("Reusing the pickle file {}.".format(train_file_pickle))
            indexed_training_dataset = self.read_pickle(train_file_pickle)
            self.data_indexer = self.read_pickle(data_indexer_pickle)
            self.data_indexer_fitted = True

        training_dataset_size = len(indexed_training_dataset.features)

        # We now need to check if the user specified max_lengths for
        # the feature, and accordingly truncate or pad if applicable. If
        # max_lengths is None for a given string key, we assume that no
        # truncation is to be done and the max lengths should be read from the
        # features.
        if not pad and max_lengths:
            raise ValueError("Passed in max_lengths {}, but set pad to false. "
                             "Did you mean to do this?".format(max_lengths))

        # Get max lengths from the dataset
        dataset_max_lengths = indexed_training_dataset.max_lengths()
        logger.info("Instance max lengths {}".format(dataset_max_lengths))
        max_lengths_to_use = dataset_max_lengths
        if pad:
            # If the user set max lengths, iterate over the
            # dictionary provided and verify that they did not
            # pass any keys to truncate that are not in the feature.
            if max_lengths is not None:
                for input_dimension, length in max_lengths.items():
                    if input_dimension in dataset_max_lengths:
                        max_lengths_to_use[input_dimension] = length
                    else:
                        raise ValueError("Passed a value for the max_lengths "
                                         "that does not exist in the "
                                         "feature. Improper input length "
                                         "dimension (key) we found was {}, "
                                         "lengths dimensions in the feature "
                                         "are {}".format(
                                             input_dimension,
                                             dataset_max_lengths.keys()))
            logger.info("Padding lengths to "
                        "length: {}".format(str(max_lengths_to_use)))
        self.training_data_max_lengths = max_lengths_to_use

        # This is a hack to get the function to run the code above immediately,
        # instead of doing the standard python generator lazy-ish evaluation.
        # This is necessary to set the class variables ASAP.
        def _get_train_data_generator():
            for indexed_feature in indexed_training_dataset.features:
                # For each feature, we want to pad or truncate if applicable
                if pad:
                    indexed_feature.pad(max_lengths_to_use)
                # Now, we want to take the feature and convert it into
                # NumPy arrays suitable for training.
                inputs, labels = indexed_feature.as_training_data(mode=mode)
                yield inputs, labels
        return _get_train_data_generator, training_dataset_size

    def get_validation_data_from_file(self, filenames, max_features=None,
                                      max_lengths=None, pad=True, mode="word"):
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
        :return: output: returns a function to construct a validation data generator
            This returns a function that can be called to produce a tuple of
            (feature generator, validation_set_size). The feature generator
            outputs features as generated by the as_validation_data function
            of the underlying feature class. The validation_set_size is the number
            of features in the validation set, which can be used to initialize a
            progress bar.
        """
        logger.info("Getting validation data from {}".format(filenames))

        pickle_validation_file = 'validation_data.p'

        if (not self.check_pickle_exists(pickle_validation_file)):
            logger.info("Processing the validation data file for first time")

            validation_dataset = TextDataset.read_from_file(filenames,
                                                            self.feature_type)
            if max_features:
                logger.info("Truncating the validation dataset "
                            "to {} features".format(max_features))
                validation_dataset = validation_dataset.truncate(max_features)

            # With our fitted data indexer, we we convert the dataset
            # from string tokens to numeric int indices.
            logger.info("Indexing validation dataset with "
                        "DataIndexer fit on train data.")
            indexed_validation_dataset = validation_dataset.to_indexed_dataset(
                self.data_indexer)
            self.write_pickle(indexed_validation_dataset, pickle_validation_file)
        else:
            logger.info("Reusing the pickle file {}.".format(pickle_validation_file))
            indexed_validation_dataset = self.read_pickle(pickle_validation_file)

        validation_dataset_size = len(indexed_validation_dataset.features)

        # We now need to check if the user specified max_lengths for
        # the feature, and accordingly truncate or pad if applicable. If
        # max_lengths is None for a given string key, we assume that no
        # truncation is to be done and the max lengths should be taken from
        # the train dataset.
        if not pad and max_lengths:
            raise ValueError("Passed in max_lengths {}, but set pad to false. "
                             "Did you mean to do this?".format(max_lengths))
        if pad:
            # Get max lengths from the train dataset
            training_data_max_lengths = self.training_data_max_lengths
            logger.info("Max lengths in training "
                        "data: {}".format(training_data_max_lengths))

            max_lengths_to_use = training_data_max_lengths
            # If the user set max lengths, iterate over the
            # dictionary provided and verify that they did not
            # pass any keys to truncate that are not in the feature.
            if max_lengths is not None:
                for input_dimension, length in max_lengths.items():
                    if input_dimension in training_data_max_lengths:
                        max_lengths_to_use[input_dimension] = length
                    else:
                        raise ValueError("Passed a value for the max_lengths "
                                         "that does not exist in the "
                                         "feature. Improper input length "
                                         "dimension (key) we found was {}, "
                                         "lengths dimensions in the feature "
                                         "are {}".format(
                                             input_dimension,
                                             training_data_max_lengths.keys()))
            logger.info("Padding lengths to "
                        "length: {}".format(str(max_lengths_to_use)))

        # This is a hack to get the function to run the code above immediately,
        # instead of doing the standard python generator lazy-ish evaluation.
        # This is necessary to set the class variables ASAP.
        def _get_validation_data_generator():
            for indexed_val_feature in indexed_validation_dataset.features:
                # For each feature, we want to pad or truncate if applicable
                if pad:
                    indexed_val_feature.pad(max_lengths_to_use)
                # Now, we want to take the feature and convert it into
                # NumPy arrays suitable for validation.
                inputs, labels = indexed_val_feature.as_training_data(mode=mode)

                yield inputs, labels
        return _get_validation_data_generator, validation_dataset_size

    def get_test_data_from_file(self, filenames, max_features=None,
                                max_lengths=None, pad=True, mode="word"):
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
        logger.info("Getting test data from {}".format(filenames))
        
        pickle_test_file = 'test_data.p'

        if (not self.check_pickle_exists(pickle_test_file)):
            logger.info("Processing the train data file for first time")
            
            test_dataset = TextDataset.read_from_file(filenames,
                                                      self.feature_type)
            if max_features:
                logger.info("Truncating the test dataset "
                            "to {} features".format(max_features))
                test_dataset = test_dataset.truncate(max_features)

            # With our fitted data indexer, we we convert the dataset
            # from string tokens to numeric int indices.
            logger.info("Indexing test dataset with DataIndexer "
                        "fit on train data.")
            indexed_test_dataset = test_dataset.to_indexed_dataset(
                self.data_indexer)
            self.write_pickle(indexed_test_dataset, pickle_test_file)

        else:
            logger.info("Reusing the pickle file {}.".format(pickle_test_file))
            indexed_test_dataset = self.read_pickle(pickle_test_file)

        test_dataset_size = len(indexed_test_dataset.features)

        # We now need to check if the user specified max_lengths for
        # the feature, and accordingly truncate or pad if applicable. If
        # max_lengths is None for a given string key, we assume that no
        # truncation is to be done and the max lengths should be taken from
        # the train dataset.
        if not pad and max_lengths:
            raise ValueError("Passed in max_lengths {}, but set pad to false. "
                             "Did you mean to do this?".format(max_lengths))
        if pad:
            # Get max lengths from the train dataset
            training_data_max_lengths = self.training_data_max_lengths
            logger.info("Max lengths in training "
                        "data: {}".format(training_data_max_lengths))

            max_lengths_to_use = training_data_max_lengths
            # If the user set max lengths, iterate over the
            # dictionary provided and verify that they did not
            # pass any keys to truncate that are not in the feature.
            if max_lengths is not None:
                for input_dimension, length in max_lengths.items():
                    if input_dimension in training_data_max_lengths:
                        max_lengths_to_use[input_dimension] = length
                    else:
                        raise ValueError("Passed a value for the max_lengths "
                                         "that does not exist in the "
                                         "feature. Improper input length "
                                         "dimension (key) we found was {}, "
                                         "lengths dimensions in the feature "
                                         "are {}".format(
                                             input_dimension,
                                             training_data_max_lengths.keys()))
            logger.info("Padding lengths to "
                        "length: {}".format(str(max_lengths_to_use)))

        # This is a hack to get the function to run the code above immediately,
        # instead of doing the standard python generator lazy-ish evaluation.
        # This is necessary to set the class variables ASAP.
        def _get_test_data_generator():
            for indexed_test_feature in indexed_test_dataset.features:
                # For each feature, we want to pad or truncate if applicable
                if pad:
                    indexed_test_feature.pad(max_lengths_to_use)
                # Now, we want to take the feature and convert it into
                # NumPy arrays suitable for validation.
                inputs = indexed_test_feature.as_testing_data(mode=mode)
                yield inputs

        return _get_test_data_generator, test_dataset_size