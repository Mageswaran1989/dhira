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
                 vocabulary_size = 5000,
                 min_occurrences = 2,
                 left_margin_size = 2,
                 right_margin_size = 2,
                 name='GloveDataset',
                 feature_type = GloveFeature,
                 train_files = None,
                 test_files = None,
                 val_files = None):
        """
        1. Convert the lines in text file as tokens with NLP.
        2. Make pairs of  (left_context, word, right_context) for given margin
        3. Filter words based on their occurrences in the whole corpus
        4. Make a map of word -> index
        5. Construct cooccurrence_matrix with (word_index, context_index) : count 
        :param vocabulary_size: Number of word to be to vectorized
        :param min_occurrences: Minimum number of occurances for the word to be considered
        :param left_margin_size: Left Margin Size
        :param right_margin_size: Right Margin Size
        :param name: Name of the Model
        :param feature_type: 
        :param train_files: Any Text File Eg: Wiki Pages
        :param test_files: Any Text File Eg: Wiki Pages
        :param val_files:  Any Text File Eg: Wiki Pages
        """

        super(GloveDataset, self).__init__(name=name,
                                           feature_type=feature_type,
                                           train_files=train_files,
                                           test_files=test_files,
                                           val_files=val_files)
        self.words = None
        self.word_to_id = None
        self.cooccurrence_matrix = None

        self.vocabulary_size = vocabulary_size
        self.min_occurrences = min_occurrences
        self.left_margin_size = left_margin_size
        self.right_margin_size = right_margin_size


    def window(self, tokens, start_index, end_index):
        """
        Returns the list of words starting from `start_index`, going to `end_index`
        taken from tokens. If `start_index` is a negative number, or if `end_index`
        is greater than the index of the last word in tokens, this function will pad
        its return value with `NULL_WORD`.
        """
        last_index = len(tokens) + 1
        selected_tokens = tokens[max(start_index, 0):min(end_index, last_index) + 1]
        return selected_tokens

    def context_windows(self, tokens, left_margin_size, right_margin_size):
        """
        
        :param tokens: List of tokens
        :param left_margin_size: 
        :param right_margin_size: 
        :return: 
        """
        for i, word in enumerate(tokens):
            start_index = i - left_margin_size
            end_index = i + right_margin_size
            left_context = self.window(tokens, start_index, i - 1)
            right_context = self.window(tokens, i + 1, end_index)
            yield (left_context, word, right_context)

    def fit_to_corpus(self, lines):
        word_counts = Counter()
        cooccurrence_counts = defaultdict(float)

        # Page Number: 2
        #Let this be X
        # X_ij tabulates number of times word j (word[1]) occurs in context of word i (word[0])
        #Let X_i = Sum of X_ik for all k,  be the number of times any word appears in the context of word i.
        #Probability P_ij = P(j|i) = X_ij/X_i be the probability that word j appear in the context of word i.

        #Page Number: 3
        #The above argument suggests that the appropriate
        # starting point for word vector learning should
        # be with ratios of co-occurrence probabilities rather
        # than the probabilities themselves. Noting that the
        # ratio Pik /Pjk depends on three words i, j, and k,
        # the most general model takes the form,....

        # Let X_i = Sum of X_ik for all k,  be the number of times any word appears in the context of word i.
        #Probability P_ik = P(k|i) = X_ik/X_i be the probability that word k appear in the context of word i.

        # Let X_j = Sum of X_jk for all k,  be the number of times any word appears in the context of word j.
        #Probability P_jk = P(k|j) = X_jk/X_j be the probability that word k appear in the context of word j.
        for tokens in tqdm(lines):
            word_counts.update(tokens)
            for left_context, word_k, right_context in self.context_windows(tokens, self.left_margin_size, self.right_margin_size):
                # add (1 / distance from focal word) for this pair
                for i, context_word_i in enumerate(left_context[::-1]):
                    cooccurrence_counts[(word_k, context_word_i)] += 1 / (i + 1) # 1 is added since index is from 0
                for j, context_word_j in enumerate(right_context):
                    cooccurrence_counts[(word_k, context_word_j)] += 1 / (j + 1)

        if len(cooccurrence_counts) == 0:
            raise ValueError("No coccurrences in lines. Did you try to reuse a generator?")

        self.words = [word for word, count in word_counts.most_common(self.vocabulary_size)
                        if count >= self.min_occurrences]
        self.word_to_id = {word: i for i, word in enumerate(self.words)}


        self.cooccurrence_matrix = {
            (self.word_to_id[words[0]], self.word_to_id[words[1]]): count
            for words, count in cooccurrence_counts.items()
            if words[0] in self.word_to_id and words[1] in self.word_to_id}

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

        logger.info("Fitting the lines to get Coocurrance Matrix")

        lines = [nltk.wordpunct_tokenize(line.lower()) for line in tqdm(lines)]

        self.fit_to_corpus(lines)

        logger.info('Extracting the features...')
        features = []
        for  word_ids, counts in tqdm(self.cooccurrence_matrix.items()):
            i_indices = word_ids[0]
            j_indices = word_ids[1]
            feature = GloveFeature(i_indices, j_indices, counts)
            features.append(feature)
        print(features[:10])

        return features

    def load_train_features(self):
        self.train_features = self.read_from_file(self.train_files)

    def load_val_features(self):
        self.val_features = self.read_from_file(self.val_files)

    def load_test_features(self):
        raise NotImplementedError

    def embedding_for(self, word_str_or_id, embeddings):
        if isinstance(word_str_or_id, str):
            return embeddings[self.word_to_id[word_str_or_id]]
        elif isinstance(word_str_or_id, int):
            return embeddings[word_str_or_id]
    # TODO https://github.com/shashankg7/glove-tensorflow/blob/master/glove/utils.py