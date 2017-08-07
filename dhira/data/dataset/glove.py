import codecs
import logging
from collections import Counter, defaultdict

from tqdm import tqdm

from dhira.data.dataset.internal.dataset_base import Dataset
from dhira.data.features.glove_feature import GloveFeature

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
import nltk

class NotFitToCorpusError(Exception):
    pass

class GloveDataset(Dataset):

    def __init__(self,
                 vocabulary_size,
                 min_occurrences,
                 left_size,
                 right_size,
                 name='default',
                 feature_type = None,
                 train_files = None,
                 test_files = None,
                 val_files = None,
                 pickle_dir=None,
                 train_features=None,
                 val_features=None,
                 test_features=None):

        super(GloveDataset, self).__init__(name=name,
                                           feature_type=feature_type,
                                           train_files=train_files,
                                           test_files=test_files,
                                           val_files=val_files,
                                           pickle_dir=pickle_dir,
                                           train_features=train_features,
                                           val_features=val_features,
                                           test_features=test_features)
        self.words = None
        self.word_to_id = None
        self.cooccurrence_matrix = None

        self.vocabulary_size = vocabulary_size
        self.min_occurrences = min_occurrences
        self.left_size = left_size
        self.right_size = right_size

    def read_from_file(self, file_names):
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
        return self.read_from_lines(lines)

    def read_from_lines(self, lines):
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
                    "list of lines.".format(self.feature_type))

        # features = [feature_class.read_from_line(line) for line in tqdm(lines)]

        logger.info("Fitting the corpus to get Coocurrance Matrix")

        lines = [nltk.wordpunct_tokenize(line.lower()) for line in tqdm(lines)]

        self.fit_to_corpus(lines)

        # cooccurrences = [(word_ids[0], word_ids[1], count)
        #                  for word_ids, count in self.cooccurrence_matrix.items()]

        logger.info('Extracting the features...')
        features = []
        for  word_ids, counts in tqdm(self.cooccurrence_matrix.items()):
            i_indices = word_ids[0]
            j_indices = word_ids[1]
            feature = GloveFeature(i_indices, j_indices, counts)
            features.append(feature)

        return features


    def window(self, region, start_index, end_index):
        """
        Returns the list of words starting from `start_index`, going to `end_index`
        taken from region. If `start_index` is a negative number, or if `end_index`
        is greater than the index of the last word in region, this function will pad
        its return value with `NULL_WORD`.
        """
        last_index = len(region) + 1
        selected_tokens = region[max(start_index, 0):min(end_index, last_index) + 1]
        return selected_tokens

    def context_windows(self, region, left_size, right_size):
        for i, word in enumerate(region):
            start_index = i - left_size
            end_index = i + right_size
            left_context = self.window(region, start_index, i - 1)
            right_context = self.window(region, i + 1, end_index)
            yield (left_context, word, right_context)

    def fit_to_corpus(self, corpus):

        # self.vocabulary_size, self.min_occurrences,
        # self.left_size, self.right_size

        word_counts = Counter()
        cooccurrence_counts = defaultdict(float)
        for region in tqdm(corpus):
            word_counts.update(region)
            for l_context, word, r_context in self.context_windows(region, self.left_size, self.right_size):
                for i, context_word in enumerate(l_context[::-1]):
                    # add (1 / distance from focal word) for this pair
                    cooccurrence_counts[(word, context_word)] += 1 / (i + 1)
                for i, context_word in enumerate(r_context):
                    cooccurrence_counts[(word, context_word)] += 1 / (i + 1)

        if len(cooccurrence_counts) == 0:
            raise ValueError("No coccurrences in corpus. Did you try to reuse a generator?")

        self.words = [word for word, count in word_counts.most_common(self.vocabulary_size)
                        if count >= self.min_occurrences]
        self.word_to_id = {word: i for i, word in enumerate(self.words)}

        self.cooccurrence_matrix = {
            (self.word_to_id[words[0]], self.word_to_id[words[1]]): count
            for words, count in cooccurrence_counts.items()
            if words[0] in self.word_to_id and words[1] in self.word_to_id}


    def load_train_features(self):
        self.train_features = self.read_from_file(self.train_files)

    def load_val_features(self):
        self.val_features = self.read_from_file(self.val_files)

    def load_test_features(self):
        raise NotImplementedError