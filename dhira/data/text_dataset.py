import codecs
import itertools
import logging
from tqdm import tqdm

from .dataset import Dataset

from dhira.data.data_indexer import DataIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class TextDataset(Dataset):
    """
    A Dataset of TextFeatures, with a few helper methods. TextFeatures aren't
    useful for much until they've been indexed. So this class just has methods
    to read in data from a file and converting it into other kinds of Datasets.
    """

    def __init__(self, feature_type = None, train_files = None, test_files = None, val_files = None,
                 pickle_dir = None):
        """
        Construct a new TextDataset
        :param features: List of TextFeature
                A list of TextFeatures to construct
                    the TextDataset from.
        """
        super(TextDataset, self).__init__(feature_type=feature_type, train_files=train_files,
                                          test_files = test_files, val_files=val_files, pickle_dir=pickle_dir)

    @staticmethod
    def fit_word_dictionary(dataset: Dataset, min_count: int):
        data_indexer: DataIndexer = DataIndexer()
        data_indexer.fit_word_dictionary(dataset, min_count)
        return data_indexer

    @staticmethod
    def to_indexed_features(text_features, data_indexer):
        """
        Converts the Dataset into an IndexedDataset, given a DataIndexer.
        :param data_indexer: DataIndexer
                        The DataIndexer to use in converting words to indices.
        :return: IndexedDataset
        """
        indexed_features = [feature.to_indexed_feature(data_indexer) for
                            feature in tqdm(text_features)]
        return indexed_features



# ------------------------------------------------------------------------------

class IndexedDataset(Dataset):
    """
    A Dataset of IndexedFeatures, with some helper methods.
    IndexedFeatures have text sequences replaced with lists of word indices,
    and are thus able to be padded to consistent lengths and converted to
    training inputs.
    """

    def __init__(self, name='default', feature_type = None, train_files = None, test_files = None, val_files = None,
                 min_count=1, pad=False, max_lengths=None, index_mode='word', pickle_dir=None):
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
        super(IndexedDataset, self).__init__(name=name, feature_type=feature_type, train_files=train_files,
                                             test_files=test_files, val_files=val_files,
                                             pickle_dir=pickle_dir)
        self.min_count = min_count
        self.data_indexer = None
        self.data_indexer_fitted = False
        self.pad = pad
        self.max_lengths = max_lengths
        self.index_mode = index_mode


    def max_lengths(self):
        """

        :return: 
        """
        max_lengths = {}
        lengths = [feature.get_lengths() for feature in self.features]
        if not lengths:
            return max_lengths
        for key in lengths[0]:
            max_lengths[key] = max(x[key] if key in x else 0 for x in lengths)
        return max_lengths

    def pad_features(self, max_lengths=None):
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
        feature_max_lengths = self.max_lengths()
        logger.info("Feature max lengths: %s", str(feature_max_lengths))
        lengths_to_use = {}
        for key in feature_max_lengths:
            if max_lengths and max_lengths[key] is not None:
                lengths_to_use[key] = max_lengths[key]
            else:
                lengths_to_use[key] = feature_max_lengths[key]

        logger.info("Now actually padding features to length: %s",
                    str(lengths_to_use))
        for feature in tqdm(self.features):
            feature.pad(lengths_to_use)

    def as_training_data(self, mode="word"):
        """
        Takes each IndexedFeature and converts it into (inputs, labels),
        according to the Feature's as_training_data() method. Note that
        you might need to call numpy.asarray() on the results of this; we
        don't do that for you, because the inputs might be complicated.
        :param mode: str, optional (default="word")
            String describing whether to return the word-level representations,
            character-level representations, or both. One of "word",
            "character", or "word+character"
        :return: 
        """
        inputs = []
        labels = []
        features = self.features
        for feature in features:
            feature_inputs, label = feature.as_training_data(mode=mode)
            inputs.append(feature_inputs)
            labels.append(label)
        return inputs, labels

    def as_testing_data(self, mode="word"):
        """
        Takes each IndexedFeature and converts it into inputs,
        according to the Feature's as_testing_data() method. Note that
        you might need to call numpy.asarray() on the results of this; we
        don't do that for you, because the inputs might be complicated.
        :param mode: str, optional (default="word")
            String describing whether to return the word-level representations,
            character-level representations, or both. One of "word",
            "character", or "word+character"
        :return: 
        """
        inputs = []
        features = self.features
        for feature in features:
            feature_inputs, _ = feature.as_testing_data(mode=mode)
            inputs.append(feature_inputs)
        return inputs, []

    def sort(self, reverse=True):
        """
        Sorts the list of IndexedFeatures, in either ascending or descending order,
        if the features are IndexedSTSFeatures
        :param reverse: boolean, optional (default=True)
            Boolean which detrmines what reverse parameter is used in the
            sorting function.
        :return: 
        """
        self.features.sort(reverse=reverse)


    def read_train_data_from_file(self):

        if self.data_indexer_fitted:
            raise ValueError("You have already called get_train_data for this "
                             "dataset, so you cannnot do it again. "
                             "If you want to train on multiple datasets, pass "
                             "in a list of files.")

        logger.info("Getting training data from {}".format(self.train_files))

        if not self.check_pickle_exists(self.train_pickle_file):
            logger.info("Processing the train data file for first time")

            print("======================================")
            print(self.train_files, self.feature_type)
            dataset = TextDataset.read_from_file(self.train_files, self.feature_type)
            self.data_indexer = TextDataset.fit_word_dictionary(dataset, self.min_count)
            self.data_indexer_fitted = True
            features = dataset.to_indexed_features(self.data_indexer)

            # We now need to check if the user specified max_lengths for
            # the feature, and accordingly truncate or pad if applicable. If
            # max_lengths is None for a given string key, we assume that no
            # truncation is to be done and the max lengths should be read from the
            # features.
            if not self.pad and self.max_lengths:
                raise ValueError("Passed in max_lengths {}, but set pad to false. "
                                 "Did you mean to do this?".format(self.max_lengths))

            # Get max lengths from the dataset
            dataset_max_lengths = self.max_lengths()
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
                                             "are {}".format(
                                input_dimension,
                                dataset_max_lengths.keys()))
                logger.info("Padding lengths to "
                            "length: {}".format(str(max_lengths_to_use)))
            self.training_data_max_lengths = max_lengths_to_use

            self.train_features = [feature.pad(max_lengths_to_use) for feature in features]
            self.write_pickle(self.train_features, self.train_pickle_file)
        else:
            logger.info("Reusing the pickle file {}.".format(self.train_pickle_file))
            self.train_features = self.read_pickle(self.train_pickle_file)

    def read_val_data_from_file(self):
        logger.info("Getting validation data from {}".format(self.val_files))

        if not self.check_pickle_exists(self.val_pickle_file):
            logger.info("Processing the validation data file for first time")
            dataset = TextDataset.read_from_file(self.val_files, self.feature_type)
            features = dataset.to_indexed_features(self.data_indexer)

            # We now need to check if the user specified max_lengths for
            # the feature, and accordingly truncate or pad if applicable. If
            # max_lengths is None for a given string key, we assume that no
            # truncation is to be done and the max lengths should be read from the
            # features.
            if not self.pad and self.max_lengths:
                raise ValueError("Passed in max_lengths {}, but set pad to false. "
                                 "Did you mean to do this?".format(self.max_lengths))

            # Get max lengths from the dataset
            dataset_max_lengths = self.max_lengths()
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
                                             "are {}".format(
                                input_dimension,
                                dataset_max_lengths.keys()))
                logger.info("Padding lengths to "
                            "length: {}".format(str(max_lengths_to_use)))
            self.training_data_max_lengths = max_lengths_to_use

            self.val_features = [feature.pad(max_lengths_to_use) for feature in features]
            self.write_pickle(self.val_features, self.val_pickle_file)
        else:
            logger.info("Reusing the pickle file {}.".format(self.val_features))
            self.val_features = self.read_pickle(self.val_pickle_file)

    def read_test_data_from_file(self, filenames):
        if not self.check_pickle_exists(self.val_pickle_file):
            ''
        else:
            ''