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