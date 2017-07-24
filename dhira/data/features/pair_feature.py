import csv
from copy import deepcopy
import itertools
import numpy as np
from overrides import overrides

from .feature import TextFeature, IndexedFeature
from .feature_word import IndexedFeatureWord


class PairFeature(TextFeature):
    """
    PairFeature contains a labeled pair of sentences and one binary label.

    You could have the label represent whatever you want, in this repo the
    label indicates whether or not the sentences are duplicate questions in the
    Kaggle Quora dataset. 0 indicates that they are not duplicates, 1 indicates
    that they are.

    :param first_sentence: str
        A string of the first sentence in this training instance.

    :param second_sentence: str
        A string of the second sentence in this training instance.

    :param label: int
        An int, where 0 indicates that the two sentences are not
        duplicate questions and 1 indicates that they are.

    """

    label_mapping = {0: [1, 0], 1: [0, 1], None: None}

    def __init__(self, first_sentence, second_sentence, label):
        super(PairFeature, self).__init__(label)
        self.first_sentence_str = first_sentence
        self.second_sentence_str = second_sentence
        # Tokenize the string representations of the first
        # and second sentence into words and characters

        first_sentence_words = self.tokenizer.tokenize(first_sentence)
        second_sentence_words = self.tokenizer.tokenize(second_sentence)

        self.first_sentence_tokenized = {
            "words": first_sentence_words,
            "characters": list(map(list, first_sentence_words))
        }
        self.second_sentence_tokenized = {
            "words": second_sentence_words,
            "characters": list(map(list, second_sentence_words))
        }

    def __str__(self):
        return ('PairFeature(' + self.first_sentence_str + ', ' +
                self.second_sentence_str + ', ' + str(self.label) + ')')

    @overrides
    def words(self):
        words = deepcopy(self.first_sentence_tokenized)
        second_sentence_words = deepcopy(self.second_sentence_tokenized)

        # Flatten the list of lists of characters
        words["characters"] = list(itertools.chain.from_iterable(words["characters"]))
        second_sentence_words["characters"] = list(itertools.chain.from_iterable(
            second_sentence_words["characters"]))

        for namespace in words:
            words[namespace].extend(second_sentence_words[namespace])
        return words

    @overrides
    def to_indexed_feature(self, data_indexer):
        indexed_first_words, indexed_first_chars = self._index_text(
            self.first_sentence_tokenized,
            data_indexer)
        indexed_second_words, indexed_second_chars = self._index_text(
            self.second_sentence_tokenized,
            data_indexer)
        # These are lists of IndexedFeatureWords
        indexed_first_sentence = [IndexedFeatureWord(word, word_characters) for
                                  word, word_characters in zip(indexed_first_words,
                                                               indexed_first_chars)]
        indexed_second_sentence = [IndexedFeatureWord(word, word_characters) for
                                   word, word_characters in zip(indexed_second_words,
                                                                indexed_second_chars)]

        return IndexedPairFeature(indexed_first_sentence,
                                  indexed_second_sentence,
                                  self.label_mapping[self.label])

    @classmethod
    def read_from_line(cls, line):
        """
        Given a string line from the dataset, construct an PairFeature from it.

        :param line: str
            The line from the dataset from which to construct an PairFeature
            from. Expected line format for training data:
            (1) [id],[qid1],[qid2],[question1],[question2],[is_duplicate]
            Or, in the case of the test set:
            (2) [id],[question1],[question2]

        :return instance: PairFeature
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
        return cls(first_sentence, second_sentence, label)

#--------------------------------------------------------------------------------

class IndexedPairFeature(IndexedFeature):
    """
    This is an indexed instance that is commonly used for sentence
    pairs with a label. In this repo, we are using it to indicate
    the indices of two question sentences, and the label is a one-hot
    vector indicating whether the two sentences are duplicates.

    :param first_sentence_indices: List of IndexedFeatureWord
        A list of IndexedFeatureWord representing the word and character
        indices of the first sentence.

    :param second_sentence_indices: List of IndexedFeatureWord
        A list of IndexedFeatureWord representing the word and character
        indices of the second sentence.

    :param label: List of int
        A list of integers representing the label. If the two sentences
        are not duplicates, the indexed label is [1, 0]. If the two sentences
        are duplicates, the indexed label is [0, 1].
    """
    def __init__(self, first_sentence_indices, second_sentence_indices, label):
        super(IndexedPairFeature, self).__init__(label)
        self.first_sentence_indices = first_sentence_indices
        self.second_sentence_indices = second_sentence_indices

    def get_int_word_indices(self):
        """
        The method extracts the indices corresponding to the words in this
        instance.

        :return word_indices: tuple of List[int]
            A tuple of List[int], where the first list refers to the indices of the words
            in the first sentence and the second list refers to the indices of the words
            in the second sentence.
        """
        first_sentence_word_indices = [idxd_word.word_index for idxd_word in
                                       self.first_sentence_indices]
        second_sentence_word_indices = [idxd_word.word_index for idxd_word in
                                        self.second_sentence_indices]
        return (first_sentence_word_indices, second_sentence_word_indices)

    def get_int_char_indices(self):
        """
        The method extracts the indices corresponding to the characters for
        each word in this instance.

        :return char_indices: tuple of List[List[int]]
            A tuple of List[int], where the first list refers to the indices of
            the characters of the words in the first sentence. Each inner list
            returned contains the character indices, and the length of the list
            returned corresponds to the number of words in the sentence. The
            second list refers to the indices of the characters of
            the words in the second sentence.
        """
        first_sentence_char_indices = [idxd_word.char_indices for idxd_word in
                                       self.first_sentence_indices]
        second_sentence_char_indices = [idxd_word.char_indices for idxd_word in
                                        self.second_sentence_indices]
        return (first_sentence_char_indices, second_sentence_char_indices)

    @classmethod
    @overrides
    def empty_feature(cls):
        return IndexedPairFeature([], [], label=None)

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
        first_sentence_word_len = len(self.first_sentence_indices)
        second_sentence_word_len = len(self.second_sentence_indices)
        # Length of longest word, as measured in # characters
        # The length of the list can be 0, so we have to take some
        # precautions with the max.
        first_sentence_chars = [len(idxd_word.char_indices) for
                                idxd_word in self.first_sentence_indices]
        if first_sentence_chars:
            first_sentence_chars_len = max(first_sentence_chars)
        else:
            first_sentence_chars_len = 0
        second_sentence_chars = [len(idxd_word.char_indices) for
                                 idxd_word in self.second_sentence_indices]
        if second_sentence_chars:
            second_sentence_chars_len = max(second_sentence_chars)
        else:
            second_sentence_chars_len = 0
        lengths = {"num_sentence_words": max(first_sentence_word_len,
                                             second_sentence_word_len),
                   "num_word_characters": max(first_sentence_chars_len,
                                              second_sentence_chars_len)}
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
        num_word_characters = max_lengths["num_word_characters"]
        # Pad at the word-level, adding empty IndexedFeatureWords
        self.first_sentence_indices = self.pad_sequence_to_length(
            self.first_sentence_indices,
            num_sentence_words,
            default_value=IndexedFeatureWord.padding_instance_word)
        self.second_sentence_indices = self.pad_sequence_to_length(
            self.second_sentence_indices,
            num_sentence_words,
            default_value=IndexedFeatureWord.padding_instance_word)

        # Pad at the character-level, adding 0 padding to character list
        for indexed_instance_word in self.first_sentence_indices:
            indexed_instance_word.char_indices = self.pad_sequence_to_length(
                indexed_instance_word.char_indices,
                num_word_characters)

        for indexed_instance_word in self.second_sentence_indices:
            indexed_instance_word.char_indices = self.pad_sequence_to_length(
                indexed_instance_word.char_indices,
                num_word_characters)

    @overrides
    def as_training_data(self, mode="word"):
        """
        Transforms the instance into a collection of NumPy
        arrays suitable for use as training data in the model.

        :param mode: str, optional (default="word")
            String describing whether to return the word-level representations,
            character-level representations, or both. One of "word",
            "character", or "word+character"

        :return data_tuple: tuple
            The outer tuple has two elements.
            The first element of this outer tuple is another tuple, with the
            "training data". In this case, this is the NumPy arrays of the
            first and second sentence. The second element of the outer tuple is
            a NumPy array with the label.
        """
        if self.label is None:
            raise ValueError("self.label is None so this is a test example. "
                             "You cannot call as_training_data on it.")
        if mode not in set(["word", "character", "word+character"]):
            raise ValueError("Input mode was {}, expected \"word\","
                             "\"character\", or \"word+character\"")
        if mode == "word" or mode == "word+character":
            first_sentence_word_array = np.asarray([word.word_index for word
                                                    in self.first_sentence_indices],
                                                   dtype="int32")
            second_sentence_word_array = np.asarray([word.word_index for word
                                                     in self.second_sentence_indices],
                                                    dtype="int32")
        if mode == "character" or mode == "word+character":
            first_sentence_char_matrix = np.asarray([word.char_indices for word
                                                     in self.first_sentence_indices],
                                                    dtype="int32")
            second_sentence_char_matrix = np.asarray([word.char_indices for word
                                                      in self.second_sentence_indices],
                                                     dtype="int32")
        if mode == "character":
            return ((first_sentence_char_matrix, second_sentence_char_matrix),
                    (np.asarray(self.label),))
        if mode == "word":
            return ((first_sentence_word_array, second_sentence_word_array),
                    (np.asarray(self.label),))
        if mode == "word+character":
            return ((first_sentence_word_array, first_sentence_char_matrix,
                     second_sentence_word_array, second_sentence_char_matrix),
                    (np.asarray(self.label),))

    @overrides
    def as_testing_data(self, mode="word"):
        """
        Transforms the instance into a collection of NumPy
        arrays suitable for use as testing data in the model.

        :return data_tuple: tuple
            The first element of this tuple has the NumPy array
            of the first sentence, and the second element has the
            NumPy array of the second sentence.

        :return mode: str, optional (default="word")
            String describing whether to return the word-level representations,
            character-level representations, or both. One of "word",
            "character", or "word+character"
        """
        if mode not in set(["word", "character", "word+character"]):
            raise ValueError("Input mode was {}, expected \"word\","
                             "\"character\", or \"word+character\"")
        if mode == "word" or mode == "word+character":
            first_sentence_word_array = np.asarray([word.word_index for word
                                                    in self.first_sentence_indices],
                                                   dtype="int32")
            second_sentence_word_array = np.asarray([word.word_index for word
                                                     in self.second_sentence_indices],
                                                    dtype="int32")
        if mode == "character" or mode == "word+character":
            first_sentence_char_matrix = np.asarray([word.char_indices for word
                                                     in self.first_sentence_indices],
                                                    dtype="int32")
            second_sentence_char_matrix = np.asarray([word.char_indices for word
                                                      in self.second_sentence_indices],
                                                     dtype="int32")
        if mode == "character":
            return ((first_sentence_char_matrix, second_sentence_char_matrix),
                    ())
        if mode == "word":
            return ((first_sentence_word_array, second_sentence_word_array),
                    ())
        if mode == "word+character":
            return ((first_sentence_word_array, first_sentence_char_matrix,
                     second_sentence_word_array, second_sentence_char_matrix),
                    ())

    @overrides
    def __eq__(self, other):
        """
        Checks for equality between this instance and another instance.
        Two IndexedPairFeature objects are equal when they have the same
        sentence lengths and the same word indices for each sentence.

        :param other: IndexedPairFeature
            The IndexedPairFeature this instance is being compared against.

        :return equality: boolean
            Returns whether or not the two instances are equal.
        """

        if not isinstance(other, self.__class__):
            return False

        this_length = self.get_lengths()["num_sentence_words"]
        other_length = other.get_lengths()["num_sentence_words"]
        if this_length == other_length:
            sentence_1, sentence_2 = self.get_int_word_indices()
            other_sentence_1, other_sentence_2 = other.get_int_word_indices()
            for word_instance_1, word_instance_2 in zip(sentence_1,
                                                        other_sentence_1):
                if word_instance_1 != word_instance_2:
                    return False
            for word_instance_1, word_instance_2 in zip(sentence_2,
                                                        other_sentence_2):
                if word_instance_1 != word_instance_2:
                    return False
            return True
        else:
            return False

    @overrides
    def __lt__(self, other):
        """
        Checks for the less than relationship between this instance
        and another instance. if the maximum length of the two sentences in this instance
        is less than the maximum length of the two sentences in the other instance.

        :param other: IndexedPairFeature
            The IndexedPairFeature this instance is being compared against.

        :return lt boolean
            Returns whether or not the this instance is less than the other.
        """
        if not isinstance(other, self.__class__):
            return False

        this_length = self.get_lengths()["num_sentence_words"]
        other_length = other.get_lengths()["num_sentence_words"]
        if this_length == other_length:
            sentence_1, sentence_2 = self.get_int_word_indices()
            other_sentence_1, other_sentence_2 = other.get_int_word_indices()
            for word_instance_1, word_instance_2 in zip(sentence_1,
                                                        other_sentence_1):
                if word_instance_1 != word_instance_2:
                    return word_instance_1 < word_instance_2
            for word_instance_1, word_instance_2 in zip(sentence_2,
                                                        other_sentence_2):
                if word_instance_1 != word_instance_2:
                    return word_instance_1 < word_instance_2
            return False
        else:
            return this_length < other_length