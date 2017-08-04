import codecs
import itertools
import logging
import os
import sys
import spacy
from tqdm import tqdm
import dhira.data.utils.time as time
from dhira.data.dataset.dataset_base import Dataset
from dhira.data.dataset.text import TextDataset
from dhira.data.features.quora_feature import QuoraFeature

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class QuoraDataset(Dataset):
    """
    A Dataset of IndexedFeatures, with some helper methods.
    IndexedFeatures have text sequences replaced with lists of word indices,
    and are thus able to be padded to consistent lengths and converted to
    training inputs.
    """

    def __init__(self, name='quora',
                 feature_type = None,
                 train_files = None,
                 test_files = None,
                 val_files = None,
                 min_count=1,
                 pad=False,
                 max_lengths=None,
                 index_mode='word',
                 pickle_dir=None,
                 train_features=None,
                 val_features=None,
                 test_features=None,
                 spacy_nlp_pipeline=None):
        """
        :param features: Use `read_from_file` method to load the features from the dataset
        :param min_count: int, default=1
            The minimum number of times a word must occur in order
            to be indexed. 
        :param pad: boolean, default=True
            If True, pads or truncates the features to either the input
            max_length or max_length across the train filenames. If False,
            no padding or truncation is applied. 
        :param max_features: int, default=None
            If not None, the maximum number of features to produce as
            training data. If necessary, we will truncate the dataset.
            Useful for debugging and making sure things work with small
            amounts of data.
        :param max_length: dict from str to int, default=None
            If not None, the max length of a sequence in a given dimension.
            The keys for this dict must be in the same format as
            the features' get_lengths() function. These are the lengths
            that the features are padded or truncated to.
        :param  mode: str, optional (default="word")
            String describing whether to return the word-level representations,
            character-level representations, or both. One of "word",
            "character", or "word+character"
        :param spacy_nlp_pipeline Eg: spacy.load('en_core_web_md')
        """
        super(QuoraDataset, self).__init__(name=name,
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

        self.data_indexer = None
        self.data_indexer_fitted = False
        self.nlp = spacy_nlp_pipeline


        # We now need to check if the user specified max_length for
        # the feature, and accordingly truncate or pad if applicable. If
        # max_length is None for a given string key, we assume that no
        # truncation is to be done and the max lengths should be read from the
        # features.
        if not self.pad and self.max_lengths:
            raise ValueError("Passed in max_length {}, but set pad to false. "
                             "Did you mean to do this?".format(self.max_lengths))

        logger.info('Trying to load prefitted data_indexer... {}'.format(self.indexer_pickle_file))
        if self.check_pickle_exists(self.indexer_pickle_file):
            logger.info("Reusing the pickle file {}.".format(self.indexer_pickle_file))
            self.data_indexer = self.read_pickle(self.indexer_pickle_file)
            self.data_indexer_fitted = True
        else:
            logger.info("Data Indexer needs to be fitted...")


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
        feature_max_lengths = QuoraDataset.max_lengths(features)
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

    def to_features(self, file_names, feature_type):
        """
        Read a dataset (basically a list of features) from
        a data file.
        :param file_names: str or List of str
                 The string filename from which to read the features, or a List of
            strings repesenting the files to pull features from. If a string
            is passed in, it is automatically converted to a single-element
            list.
        :param feature_class: Feature
            The Feature class to create from these lines.
        :return: text_dataset: TextDataset
            A new TextDataset with the features read from the file.
        """

        if isinstance(file_names, str):
            file_names = [file_names]
        # If file_names is not a list, throw an error. If it is a list,
        # but the first element isn't a string, also throw an error.
        if not isinstance(file_names, list) or not isinstance(file_names[0],
                                                              str):
            raise ValueError("Expected filename to be a List of strings "
                             "but was {} of type "
                             "{}".format(file_names, type(file_names)))
        logger.info("Reading files {} to a list of lines.".format(file_names))
        lines = [x.strip() for filename in file_names
                 for x in tqdm(codecs.open(filename, "r", "utf-8").readlines())]
        return self.lines_to_features(lines, feature_type)

    def lines_to_features(self, lines, feature_type):
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
                    "list of lines.".format(feature_type))

        features = [feature_type.read_from_line(line, self.nlp) for line in tqdm(lines)]

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

        return features

    def max_lengths_to_use(self, features):
        # Get max lengths from the dataset
        dataset_max_lengths = QuoraDataset.max_lengths(features)
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

    def load_train_features_from_file(self):

        logger.info("Getting training data from {}".format(self.train_files))

        if not self.check_pickle_exists(self.train_pickle_file) and \
                not self.check_pickle_exists(self.indexer_pickle_file): #TODO remove this line
            logger.info("Processing the train data file for first time")
            self.train_features = self.to_features(self.train_files, self.feature_type)
            self.data_indexer = TextDataset.fit_data_indexer(self.train_features, self.min_count)
            self.data_indexer_fitted = True
            self.train_features = TextDataset.to_indexed_features(self.train_features, self.data_indexer)
            self.training_data_max_lengths = self.max_lengths_to_use(self.train_features)

            self.write_pickle(self.train_features, self.train_pickle_file)
            self.write_pickle(self.data_indexer, self.indexer_pickle_file)
        else:
            logger.info("Reusing the pickle file {}.".format(self.train_pickle_file))
            self.train_features = self.read_pickle(self.train_pickle_file)
            self.training_data_max_lengths = self.max_lengths_to_use(self.train_features)
            # logger.info("Reusing the pickle file {}.".format(self.indexer_pickle_file))
            # self.data_indexer = self.read_pickle(self.indexer_pickle_file)



    def load_val_features_from_file(self):
        logger.info("Getting validation data from {}".format(self.val_files))

        if not self.check_pickle_exists(self.val_pickle_file):
            logger.info("Processing the validation data file for first time")
            self.val_features = self.to_features(self.val_files, self.feature_type)
            self.val_features = TextDataset.to_indexed_features(self.val_features, self.data_indexer)


            self.write_pickle(self.val_features, self.val_pickle_file)
        else:
            logger.info("Reusing the pickle file {}.".format(self.val_features))
            self.val_features = self.read_pickle(self.val_pickle_file)

            self.max_lengths_to_use(self.val_features)

    def load_test_features_from_file(self):
        logger.info("Getting test data from {}".format(self.test_files))

        if not self.check_pickle_exists(self.test_pickle_file):
            logger.info("Processing the test data file for first time")
            self.test_features = self.to_features(self.test_files, self.feature_type)
            self.test_features = TextDataset.to_indexed_features(self.test_features, self.data_indexer)

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
            inputs, labels = feature.as_training_data()
            yield inputs, labels

    def get_validation_batch_generator(self):
        for feature in self.val_features:
            # For each instance, we want to pad or truncate if applicable
            if self.pad:
                feature.pad(self.training_data_max_lengths)
            # Now, we want to take the instance and convert it into
            # NumPy arrays suitable for training.
            inputs, labels = feature.as_training_data()
            yield inputs, labels

    def get_test_batch_generator(self):
        for feature in self.test_features:
            # For each instance, we want to pad or truncate if applicable
            if self.pad:
                feature.pad(self.training_data_max_lengths)
            # Now, we want to take the instance and convert it into
            # NumPy arrays suitable for training.
            inputs, labels = feature.as_training_data()
            yield inputs, labels

    def custom_input(self, kwargs):
        """
        Takes two questions and converts them into `QuoraFeatureIndexed` feature
        :param kwargs: dict
                    question_1: str
                    question_2: str
        :param nlp spaCy nlp pipeline
        :return: Batch of QuoraFeatureIndexed to be processed with predict
        """
        first_sentence_token = QuoraFeature.get_tokens(kwargs['question_1'], self.nlp)
        second_sentence_token = QuoraFeature.get_tokens(kwargs['question_2'], self.nlp)
        single_feature = QuoraFeature(first_sentence_token,second_sentence_token, None).to_indexed_feature(self.data_indexer)
        return single_feature