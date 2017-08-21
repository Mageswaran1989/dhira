import codecs
import itertools
import logging

from tqdm import tqdm

from dhira.data.dataset.internal.text import TextDataset
from dhira.data.features.quora_feature import QuoraFeature

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class QuoraDataset(TextDataset):
    """
    A Dataset of IndexedFeatures, with some helper methods.
    IndexedFeatures have text sequences replaced with lists of word indices,
    and are thus able to be padded to consistent lengths and converted to
    training inputs.
    """

    def __init__(self,
                 train_files,
                 test_files,
                 val_files,
                 name='quora',
                 feature_type = QuoraFeature,
                 min_count=1,
                 pad=False,
                 max_lengths={"num_sentence_words": 30},
                 index_mode='word'):
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
                                             min_count=min_count,
                                             pad=pad,
                                             max_lengths=max_lengths,
                                             index_mode=index_mode)

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
        return self.lines_to_features(lines[:100], feature_type)

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


    def _load_train_features(self):
        logger.info("Getting training data from {}".format(self.train_files))
        self.train_features = self.to_features(self.train_files, self.feature_type)
        # self.training_data_max_lengths = self.max_lengths_to_use(self.train_features)

    def _load_val_features(self):
        logger.info("Getting validation data from {}".format(self.val_files))
        self.val_features = self.to_features(self.val_files, self.feature_type)

        # self.max_lengths_to_use(self.val_features)

    def _load_test_features(self):
        logger.info("Getting test data from {}".format(self.test_files))
        self.test_features = self.to_features(self.test_files, self.feature_type)

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

    def custom_input(self, question_1, question_2):
        """
        Takes two questions and converts them into `QuoraFeatureIndexed` feature
        :param question_1: 
        :param question_2: 
        :return: QuoraFeatureIndexed to be batched with  DataManager.to_batch()
        """

        first_sentence_token = self.feature_type.tokenize(question_1, self.nlp)
        second_sentence_token = self.feature_type.tokenize(question_2, self.nlp)
        single_feature = self.feature_type(first_sentence_token,second_sentence_token, None).\
            to_indexed_feature(self.data_indexer)
        return single_feature