import logging

from tqdm import tqdm_notebook as tqdm

from dhira.data.dataset.internal.dataset_base import Dataset
from dhira.data.dataset.internal.text import TextDataset

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class IndexedDataset(Dataset):
    """
    A Dataset of IndexedFeatures, with some helper methods.
    IndexedFeatures have text sequences replaced with lists of word indices,
    and are thus able to be padded to consistent lengths and converted to
    training inputs.
    """

    def __init__(self, name='default', feature_type = None, train_files = None, test_files = None, val_files = None,
                 min_count=1, pad=False, max_lengths=None, index_mode='word', pickle_dir=None, train_features=None,
                 val_features=None,
                 test_features=None):
        """
        :param features: Use `read_from_file` method to load the features from the dataset
        :param min_count: int, default=1
            The minimum number of times a word must occur in order
            to be indexed. 
        :param pad: boolean, default=True
            If True, pads or truncates the features to either the input
            max_lengths or max_lengths across the train filenames. If False,
            no padding or truncation is applied. 
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
        :param  mode: str, optional (default="word")
            String describing whether to return the word-level representations,
            character-level representations, or both. One of "word",
            "character", or "word+character"
        """
        super(IndexedDataset, self).__init__(name=name,
                                             feature_type=feature_type,
                                             train_files=train_files,
                                             test_files=test_files,
                                             val_files=val_files,
                                             pickle_dir=pickle_dir,
                                             train_features=train_features,
                                             val_features=val_features,
                                             test_features=test_features,)
        self.min_count = min_count
        self.data_indexer = None
        self.data_indexer_fitted = False
        self.pad = pad
        self.max_lengths = max_lengths
        self.index_mode = index_mode

        self.max_lengths_to_use = None

        # We now need to check if the user specified max_lengths for
        # the feature, and accordingly truncate or pad if applicable. If
        # max_lengths is None for a given string key, we assume that no
        # truncation is to be done and the max lengths should be read from the
        # features.
        if not self.pad and self.max_lengths:
            raise ValueError("Passed in max_lengths {}, but set pad to false. "
                             "Did you mean to do this?".format(self.max_lengths))

    @staticmethod
    def max_lengths(features):
        """

        :return: 
        """
        max_lengths = {}
        lengths = [feature.get_lengths() for feature in features]
        if not lengths:
            return max_lengths
        for key in lengths[0]:
            max_lengths[key] = max(x[key] if key in x else 0 for x in lengths)
        return max_lengths

    @staticmethod
    def pad_features(features, max_lengths=None):
        """
        Make all of the IndexedFeatures in the dataset have the same length
        by padding them (in the front) with zeros.
        If max_length is given for a particular dimension, we will pad all
        features to that length (including left-truncating features if
        necessary). If not, we will find the longest feature and pad all
        features to that length. Note that max_lengths is a _List_, not an int
        - there could be several dimensions on which we need to pad, depending
        on what kind of feature we are dealing with.
        This method _modifies_ the current object, it does not return a new
        IndexedDataset.
        """
        # First we need to decide _how much_ to pad. To do that, we find the
        # max length for all relevant padding decisions from the features
        # themselves. Then we check whether we were given a max length for a
        # particular dimension. If we were, we use that instead of the
        # feature-based one.
        logger.info("Getting max lengths from features")
        feature_max_lengths = IndexedDataset.max_lengths(features)
        logger.info("Feature max lengths: %s", str(feature_max_lengths))
        lengths_to_use = {}
        for key in feature_max_lengths:
            if max_lengths and max_lengths[key] is not None:
                lengths_to_use[key] = max_lengths[key]
            else:
                lengths_to_use[key] = feature_max_lengths[key]

        logger.info("Now actually padding features to length: %s",
                    str(lengths_to_use))
        for feature in tqdm(features):
            feature.pad(lengths_to_use)

        return features

    def max_lengths_to_use(self, features):
        # Get max lengths from the dataset
        dataset_max_lengths = IndexedDataset.max_lengths(features)
        logger.info("Instance max lengths {}".format(dataset_max_lengths))
        max_lengths_to_use = dataset_max_lengths
        if self.pad:
            # If the user set max lengths, iterate over the
            # dictionary provided and verify that they did not
            # pass any keys to truncate that are not in the feature.
            if self.max_lengths is not None:
                for input_dimension, length in self.max_lengths.items():
                    if input_dimension in dataset_max_lengths:
                        max_lengths_to_use[input_dimension] = length
                    else:
                        raise ValueError("Passed a value for the max_lengths "
                                         "that does not exist in the "
                                         "feature. Improper input length "
                                         "dimension (key) we found was {}, "
                                         "lengths dimensions in the feature "
                                         "are {}".format(input_dimension,
                                                         dataset_max_lengths.keys()))
            logger.info("Padding lengths to length: {}".format(str(max_lengths_to_use)))
        return max_lengths_to_use

    def load_train_features(self):

        if self.data_indexer_fitted:
            raise ValueError("You have already called get_train_data for this "
                             "dataset, so you cannnot do it again. "
                             "If you want to train on multiple datasets, pass "
                             "in a list of files.")

        logger.info("Getting training data from {}".format(self.train_files))

        if not self.check_pickle_exists(self.train_pickle_file) and \
                not self.check_pickle_exists(self.indexer_pickle_file):
            logger.info("Processing the train data file for first time")
            self.train_features = TextDataset.to_features(self.train_files, self.feature_type)
            self.data_indexer = TextDataset.fit_data_indexer(self.train_features, self.min_count)
            self.data_indexer_fitted = True
            self.train_features = TextDataset.to_indexed_features(self.train_features, self.data_indexer)
            self.training_data_max_lengths = self.max_lengths_to_use(self.train_features)

            self.write_pickle(self.train_features, self.train_pickle_file)
            self.write_pickle(self.data_indexer, self.indexer_pickle_file)
        else:
            logger.info("Reusing the pickle file {}.".format(self.train_pickle_file))
            self.train_features = self.read_pickle(self.train_pickle_file)
            logger.info("Reusing the pickle file {}.".format(self.indexer_pickle_file))
            self.data_indexer = self.read_pickle(self.indexer_pickle_file)

            self.training_data_max_lengths = self.max_lengths_to_use(self.train_features)

    def load_val_features(self):
        logger.info("Getting validation data from {}".format(self.val_files))

        if not self.check_pickle_exists(self.val_pickle_file):
            logger.info("Processing the validation data file for first time")
            self.val_features = TextDataset.to_features(self.val_files, self.feature_type)
            self.val_features = TextDataset.to_indexed_features( self.val_features, self.data_indexer)

            self.max_lengths_to_use(self.val_features)

            self.write_pickle(self.val_features, self.val_pickle_file)
        else:
            logger.info("Reusing the pickle file {}.".format(self.val_features))
            self.val_features = self.read_pickle(self.val_pickle_file)

    def load_test_features(self):
        logger.info("Getting test data from {}".format(self.test_files))

        if not self.check_pickle_exists(self.test_pickle_file):
            logger.info("Processing the test data file for first time")
            self.test_features = TextDataset.to_features(self.test_files, self.feature_type)
            self.test_features = TextDataset.to_indexed_features( self.test_features, self.data_indexer)

            self.max_lengths_to_use(self.test_features)

            self.write_pickle(self.test_features, self.test_pickle_file)
        else:
            logger.info("Reusing the pickle file {}.".format(self.test_features))
            self.test_features = self.read_pickle(self.test_pickle_file)

    def get_train_batch_generator(self):
        for feature in self.train_features:
            # For each instance, we want to pad or truncate if applicable
            if self.pad:
                feature.pad(self.training_data_max_lengths)
            # Now, we want to take the instance and convert it into
            # NumPy arrays suitable for training.
            inputs, labels = feature.as_training_data(mode='word')
            yield inputs, labels

    def get_validation_batch_generator(self):
        for feature in self.val_features:
            # For each instance, we want to pad or truncate if applicable
            if self.pad:
                feature.pad(self.training_data_max_lengths)
            # Now, we want to take the instance and convert it into
            # NumPy arrays suitable for training.
            inputs, labels = feature.as_training_data(mode='word')
            yield inputs, labels

    def get_test_batch_generator(self):
        for feature in self.test_features:
            # For each instance, we want to pad or truncate if applicable
            if self.pad:
                feature.pad(self.training_data_max_lengths)
            # Now, we want to take the instance and convert it into
            # NumPy arrays suitable for training.
            inputs, labels = feature.as_training_data(mode='word')
            yield inputs, labels