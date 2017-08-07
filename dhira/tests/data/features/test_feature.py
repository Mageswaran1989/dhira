from dhira.data.features.pair_feature import PairFeature

from dhira.data.data_indexer import DataIndexer
from dhira.data.features.internal.feature_base import IndexedFeature
from dhira.data.features.internal.feature_base import TextFeature
from dhira.data.nlp import Tokenizer
from dhira.tests.common.test_case import DuplicateTestCase


class TestTextFeature(DuplicateTestCase):
    """
    The point of this test class is to test the tokenizer used by the
    TextInstance, to be sure that we get what we expect.
    """
    def test_word_tokenizer_tokenizes_the_sentence_correctly(self):
        feature = PairFeature("One sentence.",
                               "A two sentence.", Tokenizer)
        assert feature.words() == {"words": ["one", "sentence",
                                              ".", "a", "two", "sentence", "."],
                                    "characters": ['o', 'n', 'e', 's', 'e', 'n',
                                                   't', 'e', 'n', 'c', 'e', '.',
                                                   'a', 't', 'w', 'o', 's', 'e',
                                                   'n', 't', 'e', 'n', 'c', 'e', '.']}

    def test_exceptions(self):
        feature = TextFeature()
        data_indexer = DataIndexer()
        with self.assertRaises(NotImplementedError):
            feature.words()
        with self.assertRaises(NotImplementedError):
            feature.to_indexed_feature(data_indexer)
        with self.assertRaises(RuntimeError):
            feature.read_from_line("some line")
        with self.assertRaises(NotImplementedError):
            feature.words()


class TestIndexedInstance(DuplicateTestCase):
    def test_exceptions(self):
        feature = IndexedFeature()
        with self.assertRaises(NotImplementedError):
            feature.empty_feature()
        with self.assertRaises(NotImplementedError):
            feature.get_lengths()
        with self.assertRaises(NotImplementedError):
            feature.pad({})
        with self.assertRaises(NotImplementedError):
            feature.as_training_data()
        with self.assertRaises(NotImplementedError):
            feature.as_testing_data()