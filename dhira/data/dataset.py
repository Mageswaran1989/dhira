import codecs
import itertools
import logging

from tqdm import tqdm

from .features.feature import Feature

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class Dataset:
    """
    A collection of Features. This base class has general methods that apply
    to all collections of Features. That basically is just methods that
    operate on sets, like merging and truncating.
    """

    def __init__(self, features):
        """
        Construct a dataset from a List of Features.

        :param features: List of Features to build Dataset from.
        """
        if not isinstance(features, list):
            raise ValueError("Expected features to be type "
                             "List, found {} of type "
                             "{}".format(features, type(features)))
        if not isinstance(features[0], Feature):
            raise ValueError("Expected features to be a List "
                             "of features, but the first element "
                             "of the input list was {} of type "
                             "{}".format(features[0], type(features[0])))
        self.features = features

    def merge(self, other):
        """
        Combine two datasets. If you call try to merge two Datasets of the same
        subtype, you will end up with a Dataset of the same type (i.e., calling
        IndexedDataset.merge() with another IndexedDataset will return an
        IndexedDataset). If the types differ, this method currently raises an
        error, because the underlying Feature objects are not currently type
        compatible.
        :param other: the Dataset that needs to be merged
        :return: Merged Dataset
        """
        if type(self) is type(other):
            return self.__class__(self.features + other.features)
        else:
            raise ValueError("Cannot merge datasets with different types")

    def truncate(self, max_features):
        """
        Truncate the dataset to a fixed size.
        :param max_features (int): 
         The maximum amount of features allowed in this dataset. If there
            are more features than `max_features` in this dataset, we
            return a new dataset with a random subset of size `max_features`.
            If there are fewer than `max_features` already, we just return
            self.
        :return: 
        """
        if not isinstance(max_features, int):
            raise ValueError("Expected max_features to be type "
                             "int, found {} of type "
                             "{}".format(max_features, type(max_features)))
        if max_features < 1:
            raise ValueError("max_features must be at least 1"
                             ", found {}".format(max_features))
        if len(self.features) <= max_features:
            return self
        new_features = [i for i in self.features]
        return self.__class__(new_features[:max_features])

#-------------------------------------------------------------------------------------

class TextDataset(Dataset):
    """
    A Dataset of TextFeatures, with a few helper methods. TextFeatures aren't
    useful for much until they've been indexed. So this class just has methods
    to read in data from a file and converting it into other kinds of Datasets.
    """
    def __init__(self, features):
        """
        Construct a new TextDataset
        :param features: List of TextFeature
                A list of TextFeatures to construct
                    the TextDataset from.
        """
        super(TextDataset, self).__init__(features)

    def to_indexed_dataset(self, data_indexer):
        """
        Converts the Dataset into an IndexedDataset, given a DataIndexer.
        :param data_indexer: DataIndexer
                        The DataIndexer to use in converting words to indices.
        :return: IndexedDataset
        """
        indexed_features = [feature.to_indexed_feature(data_indexer) for
                             feature in tqdm(self.features)]
        return IndexedDataset(indexed_features)

    @staticmethod
    def read_from_file(filenames, feature_class):
        """
        Read a dataset (basically a list of features) from
        a data file.
        :param filenames: str or List of str
                 The string filename from which to read the features, or a List of
            strings repesenting the files to pull features from. If a string
            is passed in, it is automatically converted to a single-element
            list.
        :param feature_class: Feature
            The Feature class to create from these lines.
        :return: text_dataset: TextDataset
            A new TextDataset with the features read from the file.
        """
        if isinstance(filenames, str):
            filenames = [filenames]
        # If filenames is not a list, throw an error. If it is a list,
        # but the first element isn't a string, also throw an error.
        if not isinstance(filenames, list) or not isinstance(filenames[0],
                                                             str):
            raise ValueError("Expected filename to be a List of strings "
                             "but was {} of type "
                             "{}".format(filenames, type(filenames)))
        logger.info("Reading files {} to a list of lines.".format(filenames))
        lines = [x.strip() for filename in filenames
                 for x in tqdm(codecs.open(filename,
                                           "r", "utf-8").readlines())]
        return TextDataset.read_from_lines(lines, feature_class)

    @staticmethod
    def read_from_lines(lines, feature_class):
        """
        Read a dataset (basically a list of features) from
        a data file.
        :param lines: List of str
            A list containing string representations of each
            line in the file.
        :param feature_class: Feature
            The Feature class to create from these lines.
        :return: text_dataset: TextDataset
            A new TextDataset with the features read from the list.
        """
        if not isinstance(lines, list):
            raise ValueError("Expected lines to be a list, "
                             "but was {} of type "
                             "{}".format(lines, type(lines)))
        if not isinstance(lines[0], str):
            raise ValueError("Expected lines to be a list of strings, "
                             "but the first element of the list was {} "
                             "of type {}".format(lines[0], type(lines[0])))
        logger.info("Creating list of {} features from "
                    "list of lines.".format(feature_class))
        features = [feature_class.read_from_line(line) for line in tqdm(lines)]
        labels = [(x.label, x) for x in features]
        labels.sort(key=lambda x: str(x[0]))
        label_counts = [(label, len([x for x in group]))
                        for label, group
                        in itertools.groupby(labels, lambda x: x[0])]
        label_count_str = str(label_counts)
        if len(label_count_str) > 100:
            label_count_str = label_count_str[:100] + '...'
        logger.info("Finished reading dataset; label counts: %s",
                    label_count_str)
        return TextDataset(features)

#------------------------------------------------------------------------------

class IndexedDataset(Dataset):
    """
    A Dataset of IndexedFeatures, with some helper methods.
    IndexedFeatures have text sequences replaced with lists of word indices,
    and are thus able to be padded to consistent lengths and converted to
    training inputs.
    """
    def __init__(self, features):
        super(IndexedDataset, self).__init__(features)

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