from overrides import overrides
import csv
import dhira.data.utils.time as time
from copy import deepcopy
from dhira.data.features.text_feature import TextFeature
from dhira.data.features.indexed_feature import  IndexedFeature
from dhira.data.data_manager import DataManager

class QuoraFeature(TextFeature):

    def __init__(self, question1_tokens, question2_tokens, label):
        super(QuoraFeature, self).__init__(label)
        self.question1_tokenized = {
            "words" : question1_tokens
        }
        self.question2_tokenized = {
            "words" : question2_tokens
        }

    @staticmethod
    def get_tokens(sentence, nlp):
        return [tok.text for tok in nlp(sentence)]

    @staticmethod
    def read_from_line(line, nlp):
        """
        Given a string line from the dataset, construct an PairFeature from it.

        :param line: str
            The line from the dataset from which to construct an PairFeature
            from. Expected line format for training data:
            (1) [id],[qid1],[qid2],[question1],[question2],[is_duplicate]
            Or, in the case of the test set:
            (2) [id],[question1],[question2]
        :param nlp spaCy pipeline eg: spacy.load('en_core_web_md')
        :return instance: QuoraFeature
            An instance constructed from the data in the line of the dataset.
        """
        fields = list(csv.reader([line]))[0]
        if len(fields) == 6:
            # training set instance
            _, _, _, first_sentence, second_sentence, label = fields
            label = int(label)
        elif len(fields) == 3:
            # test set instance
            _, first_sentence, second_sentence = fields
            label = None
        else:
            raise RuntimeError("Unrecognized line format: " + line)

        first_sentence_token = QuoraFeature.get_tokens(first_sentence, nlp)
        second_sentence_token = QuoraFeature.get_tokens(second_sentence, nlp)
        return QuoraFeature(first_sentence_token, second_sentence_token, label)

    @overrides
    def words(self):
        #Get a copy of the 1st sentence to group the features based on keys/namespace
        words = deepcopy(self.question1_tokenized)
        second_sentence_words = deepcopy(self.question2_tokenized)

        for namespace in words:
            words[namespace].extend(second_sentence_words[namespace])
        return words

    #TODO Make generic
    def index_text(self, tokenized_sentence, data_indexer):
        word_indexed_text = [data_indexer.get_word_index(word, namespace="words")
                             for word in tokenized_sentence["words"]]
        return word_indexed_text


    @overrides
    def to_indexed_feature(self, data_indexer):
        indexed_first_words = self.index_text(
            self.question1_tokenized,
            data_indexer)
        indexed_second_words = self.index_text(
            self.question2_tokenized,
            data_indexer)

        return QuoraFeatureIndexed(indexed_first_words, indexed_second_words, self.label)

class QuoraFeatureIndexed(IndexedFeature):

    _oov_token = 0
    label_mapping = {0: [1, 0], 1: [0, 1], None: None}

    def __init__(self, question_1_indices, question_2_indices, label):
        super(QuoraFeatureIndexed, self).__init__(label)
        self.question_1_indices = question_1_indices
        self.question_2_indices = question_2_indices
        self.label = self.label_mapping[label]

    @overrides
    def get_lengths(self):
        """
        Returns the maximum length of the two
        sentences as a dictionary.

        :return lengths_dict: Dictionary of string to int
            The "num_sentence_words" and "num_word_characters" keys are
            hard-coded to have the length to pad to. This is kind
            of a messy API, but I've not yet thought of
            a cleaner way to do it.
        """
        # Length of longest sentence, as measured in # words.
        first_sentence_word_len = len(self.question_1_indices)
        second_sentence_word_len = len(self.question_2_indices)

        lengths = {"num_sentence_words": max(first_sentence_word_len,
                                             second_sentence_word_len)}
        return lengths

    @overrides
    def pad(self, max_lengths):
        """
        Pads or truncates each of the sentences, according to the input lengths
        dictionary. This dictionary is usually acquired from get_lengths.

        :param max_lengths: Dictionary of string to int
            The dictionary holding the lengths to pad the sequences to.
            In this case, we pad both to the value of the
            "num_sentence_words" key.
        """
        num_sentence_words = max_lengths["num_sentence_words"]

        # Pad at the word-level, adding 0 as padding index
        self.question_1_indices = self.pad_sequence_to_length(
            self.question_1_indices,
            num_sentence_words,
            default_value=lambda: 0)
        self.question_2_indices = self.pad_sequence_to_length(
            self.question_2_indices,
            num_sentence_words,
            default_value=lambda: 0)


    @overrides
    def as_training_data(self):
        return ((self.question_1_indices, self.question_2_indices), (self.label,))

    @overrides
    def as_validation_data(self):
        return ((self.question_1_indices, self.question_2_indices), (self.label,))

    @overrides
    def as_testing_data(self):
        return ((self.question_1_indices, self.question_2_indices), (self.label,))
