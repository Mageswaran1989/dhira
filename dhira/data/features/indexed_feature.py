from dhira.data.features.internal.feature_base import FeatureBase

class IndexedFeatureWord():
    """
    A FeatureWord represents one word in an IndexedFeature, and stores its
    int word index and character-level indices (a list of ints).

    This is mostly a convenience class for doing padding on.
    """
    def __init__(self, word_index, char_indices):
        """
        :param word_index: int
            The int index representing the word.

        :param char_indices: List[str]
            A List of indices corresponding to the characters representing
            the word.
        """
        self.word_index = word_index
        self.char_indices = char_indices

    @classmethod
    def padding_instance_word(cls):
        return IndexedFeatureWord(0, [0])

class IndexedFeature(FeatureBase):
    """
    An indexed data feature has all word tokens replaced with word indices,
    along with some kind of label, suitable for input to a model. An
    ``IndexedFeature`` is created from an ``Feature`` using a
    ``DataIndexer``, and the indices here have no recoverable meaning without
    the ``DataIndexer``.

    For example, we might have the following ``Feature``:

    - ``TrueFalseFeature('Jamie is nice, Holly is mean', True, 25)``

    After being converted into an ``IndexedFeature``, we might have
    the following:
    - ``IndexedTrueFalseFeature([1, 6, 7, 1, 6, 8], True, 25)``

    This would mean that ``"Jamie"`` and ``"Holly"`` were OOV to the
    ``DataIndexer``, and the other words were given indices.
    """

    _oov_token = 0

    @staticmethod
    def tokenize(text, nlp):
        """
        This function uses a Tokenizer to output a
        list of the indices in the input string.

        :param text: str
            The label encodes the ground truth label of the Feature.
            This encoding varies across tasks and features.

        :param nlp: spaCy nlp pipeline context

        :return  word_list: List[int]
           A list of the indices, as tokenized by the
           spaCy's tokenizer.
        """
        return [tok.rank if tok.has_vector else IndexedFeature._oov_token for tok in nlp(text)]

    @classmethod
    def empty_feature(cls):
        """
        Returns an empty, unpadded instance of this class. Necessary for option
        padding in multiple choice features.
        """
        raise NotImplementedError

    def get_lengths(self):
        """
        Returns the length of this feature in all dimensions that
        require padding.

        Different kinds of features have different fields that are padded,
        such as sentence length, number of background sentences, number of
        options, etc.

        :return lengths: {str:int}
            A dict from string to integers, where the value at each string
            key is the max length to pad that dimension.
        """
        raise NotImplementedError

    def pad(self, max_length):
        """
        Add zero-padding to make each data example of equal length for use
        in the neural network.

        This modifies the current object.

        :param max_lengths: Dictionary of {str:int}
            In this dictionary, each ``str`` refers to a type of token
            (e.g. ``max_words_question``), and the corresponding ``int`` is
            the value. This dictionary must have the same dimension as was
            returned by ``get_lengths()``. We will use these lengths to pad the
            instance in all of the necessary dimensions to the given leangths.
        """
        raise NotImplementedError


    @staticmethod
    def pad_word_sequence(word_sequence,
                          sequence_length,
                          truncate_from_right=True):
        """
        Take a list of indices and pads them.

        :param word_sequence : List of int
            A list of word indices.

        :param sequence_length : int
            The length to pad all the input sequence to.

        :param truncate_from_right : bool, default=True
            If truncating the indices is necessary, this parameter dictates
            whether we do so on the left or right. Truncating from the right
            means that when we truncate, we remove the end indices first.

        :return padded_word_sequence : List of int
            A padded list of word indices.

        Notes
        -----
        The reason we truncate from the right by default for
        questions is because the core question is generally at the start, and
        we at least want to get the core query encoded even if it means that we
        lose some of the details that are provided at the end. If you want to
        truncate from the other direction, you can.
        """
        def default_pad_value():
            return 0

        padded_word_sequence = IndexedFeature.pad_sequence_to_length(
            word_sequence, sequence_length,
            default_pad_value, truncate_from_right)
        return padded_word_sequence

    @staticmethod
    def pad_sequence_to_length(sequence,
                               desired_length,
                               default_value=lambda: 0,
                               truncate_from_right=True):
        """
        Take a list of indices and pads them to the desired length.

        :param word_sequence : List of int
            A list of word indices.

        :param desired_length : int
            Maximum length of each sequence. Longer sequences
            are truncated to this length, and shorter ones are padded to it.

        :param default_value: int, default=lambda: 0
            Callable that outputs a default value (of any type) to use as
            padding values.

        :param truncate_from_right : bool, default=True
            If truncating the indices is necessary, this parameter dictates
            whether we do so on the left or right.

        :return padded_word_sequence : List of int
            A padded or truncated list of word indices.

        Notes
        -----
        The reason we truncate from the right by default is for
        cases that are questions, with long set ups. We at least want to get
        the question encoded, which is always at the end, even if we've lost
        much of the question set up. If you want to truncate from the other
        direction, you can.
        """

        if truncate_from_right:
            truncated = sequence[:desired_length]
        else:
            truncated = sequence[-desired_length:]

        if len(truncated) < desired_length:
            # If the length of the truncated sequence is less than the desired
            # length, we need to pad.
            padding_sequence = [default_value()] * (desired_length - len(truncated))

            if truncate_from_right:
                # When we truncate from the right, we add zeroes to the end.
                truncated.extend(padding_sequence)
                return truncated
            else:
                # When we do not truncate from the right, we add zeroes to the
                # front.
                padding_sequence.extend(truncated)
                return padding_sequence
        return truncated
