from dhira.data.features.feature_base import FeatureBase
from dhira.data.nlp.spacy.tokenizer import Tokenizer

class TextFeature(FeatureBase):
    """
    An ``Feature`` that has some attached text, typically either a sentence
    or a logical form.

    This is called a ``TextFeature`` because the individual tokens here are
    encoded as strings, and we can get a list of strings out when we ask what
    words show up in the feature.

    We use these kinds of features to fit a ``DataIndexer`` (i.e., deciding
    which words should be mapped to an unknown token); to use them in training
    or testing, we need to first convert them into ``IndexedFeatures``.

    In order to actually convert text into some kind of indexed sequence, we
    rely on a ``Tokenizer``.
    """
    def __init__(self, label=None, tokenizer=None):
        if tokenizer:
            self.tokenizer: Tokenizer = tokenizer()
        super(TextFeature, self).__init__(label)

    def _words_from_text(self, text):
        """
        This function uses a Tokenizer to output a
        list of the words in the input string.

        :param text: str
            The label encodes the ground truth label of the Feature.
            This encoding varies across tasks and features.

        :return  word_list: List[str]
           A list of the words, as tokenized by the
           TextFeature's tokenizer.
        """
        return self.tokenizer.get_words_for_indexer(text)

    def _index_text(self, text, data_indexer):
        """
        This function uses a Tokenizer and an input DataIndexer to convert a
        string into a list of integers representing the word indices according
        to the DataIndexer.

        :param text: str
            The label encodes the ground truth label of the Feature.
            This encoding varies across tasks and features.

        :return index_list: List[int]
           A list of the words converted to indices, as tokenized by the
           TextFeature's tokenizer and indexed by the DataIndexer.

        """
        return  self.tokenizer.index_text(text, data_indexer)

    def words(self):
        """
        Returns a list of all of the words in this instance, contained in a
        namespace dictionary.

        This is mainly used for computing word counts when fitting a word
        vocabulary on a dataset. The namespace dictionary allows you to have
        several embedding matrices with different vocab sizes, e.g., for words
        and for characters (in fact, words and characters are the only use
        cases I can think of for now, but this allows you to do other more
        crazy things if you want). You can call the namespaces whatever you
        want, but if you want the ``DataIndexer`` to work correctly without
        namespace arguments, you should use the key 'words' to represent word
        tokens.

        :return namespace : Dictionary of {str: List[str]}
            The ``str`` key refers to vocabularies, and the ``List[str]``
            should contain the tokens in that vocabulary. For example, you
            should use the key ``words`` to represent word tokens, and the
            corresponding value in the dictionary would be a list of all the
            words in the instance.
        """
        raise NotImplementedError

    def to_indexed_feature(self, data_indexer):
        """
        Converts the words in this ``Feature`` into indices using the
        ``DataIndexer``.

        :param data_indexer : DataIndexer
            ``DataIndexer`` to use in converting the ``Feature`` to
            an ``IndexedFeature``.

        :return indexed_instance : IndexedFeature
            A ``TextFeature`` that has had all of its strings converted into
            indices.
        """
        raise NotImplementedError

    @classmethod
    def read_from_line(cls, line):
        """
        Reads an instance of this type from a line.

        :param line: str
            A line from a data file.

        :return indexed_instance: IndexedFeature
            A ``TextFeature`` that has had all of its strings converted into
            indices.

        Notes
        -----
        We throw a ``RuntimeError`` here instead of a ``NotImplementedError``,
        because it's not expected that all subclasses will implement this.
        """
        raise RuntimeError("%s feature can't be "
                           "read from a line!", str(cls))