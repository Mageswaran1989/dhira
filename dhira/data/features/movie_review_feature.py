import csv
from copy import deepcopy
from overrides import overrides
from dhira.data.features.internal.text_feature import TextFeature
from dhira.data.features.indexed_feature import IndexedFeature

class MovieReviewFeature(TextFeature):
    def __init__(self, review_tokens, label):
        """
        Create a Moview review features processed by 
        dhira.data.dataset.movie_review
        :param review_tokens: list(str) User comments tokenized by internal tokeinizer 
        :param label: 1 Positive, 0 Negative
        """
        super(MovieReviewFeature, self).__init__(label)
        self.review_tokens = {
            "words" : review_tokens
        }

    @staticmethod
    def read_from_line(line, nlp):
        """
        
        :param line: (text: str, label: str)
        :param nlp: 
        :return: 
        """
        fields = list(csv.reader([line],  delimiter='\\'))[0]
        label, text = fields
        return MovieReviewFeature(MovieReviewFeature.tokenize(text, nlp), label)

    @overrides
    def words(self):
        return self.review_tokens

    @overrides
    def to_indexed_feature(self, data_indexer):
        indexed_review_tokens = self._index_text(
            self.review_tokens,
            data_indexer)

        return MovieReviewFeatureIndexed(indexed_review_tokens, self.label)

class MovieReviewFeatureIndexed(IndexedFeature):

    label_mapping = {0: [1, 0], 1: [0, 1], None: None}

    def __init__(self, indexed_review_tokens, label):
        super(MovieReviewFeatureIndexed, self).__init__(label)
        self.indexed_review_tokens = indexed_review_tokens
        self.label = self.label_mapping[label]

    @staticmethod
    def read_from_line(line, nlp):
        """
        Use this function to index the sentence directly with spaCy nlp pipeline
        You need to get the entire embedding matrix for the pipeline provided while training
        See EmbeddingManager for more.
        :param line: str label\\text
        :param nlp: 
        :return: 
        """
        fields = list(csv.reader([line], delimiter='\\'))[0]
        label, text = fields
        text_ids = IndexedFeature.tokenize(text, nlp)
        return MovieReviewFeature(text_ids, label)

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
        lengths = {"num_sentence_words": len(self.indexed_review_tokens)}
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
        self.indexed_review_tokens = self.pad_sequence_to_length(
            self.indexed_review_tokens,
            num_sentence_words,
            default_value=lambda: 0)

    @overrides
    def as_training_data(self):
        return ((self.indexed_review_tokens,), (self.label,))

    @overrides
    def as_validation_data(self):
        return ((self.indexed_review_tokens,), (self.label,))

    @overrides
    def as_testing_data(self):
        return ((self.indexed_review_tokens,), (self.label,))