from numpy.testing import assert_allclose
import numpy as np
from dhira.data.data_indexer import DataIndexer
from dhira.data.features.feature_word import IndexedFeatureWord
from dhira.data.features.pair_feature import IndexedPairFeature
from dhira.data.features.pair_feature import PairFeature

from dhira.tests.common.test_case import DuplicateTestCase


class TestPairFeature(DuplicateTestCase):
    @staticmethod
    def feature_to_line(id, question1, question2,
                         is_duplicate=None, qid1=None, qid2=None):
        if qid1 is None and qid2 is None and is_duplicate is None:
            # test set example
            line = "\"{}\",\"{}\",\"{}\"".format(id, question1,
                                                 question2)
        else:
            line = ("\"{}\",\"{}\",\"{}\","
                    "\"{}\",\"{}\",\"{}\"").format(id, qid1, qid2, question1,
                                                   question2, is_duplicate)
        return line

    def test_read_from_line_handles_test_example(self):
        question1 = "Does he enjoy playing soccer in the rain?"
        question2 = "Does he enjoy coding in the rain?"
        id = 0
        line = self.feature_to_line(id, question1, question2)
        feature = PairFeature.read_from_line(line)
        assert feature.first_sentence_str == question1
        expected_first_sentence_words = ["does", "he", "enjoy", "playing",
                                         "soccer", "in", "the", "rain", "?"]
        expected_first_sentence_chars = list(map(list, expected_first_sentence_words))
        assert feature.first_sentence_tokenized == {
            "words": expected_first_sentence_words,
            "characters": expected_first_sentence_chars
        }
        expected_second_sentence_words = ["does", "he", "enjoy", "coding",
                                          "in", "the", "rain", "?"]
        expected_second_sentence_chars = list(map(list, expected_second_sentence_words))
        assert feature.second_sentence_tokenized == {
            "words": expected_second_sentence_words,
            "characters": expected_second_sentence_chars
        }
        assert feature.second_sentence_str == question2
        assert feature.label is None

    def test_read_from_line_handles_train_example(self):
        question1 = "Does he enjoy playing soccer in the rain?"
        question2 = "Does he enjoy coding in the rain?"
        id = 0
        qid1 = 0
        qid2 = 1
        label = 0
        line = self.feature_to_line(id, question1, question2,
                                     label, qid1, qid2)
        feature = PairFeature.read_from_line(line)
        assert feature.first_sentence_str == question1
        expected_first_sentence_words = ["does", "he", "enjoy", "playing",
                                         "soccer", "in", "the", "rain", "?"]
        expected_first_sentence_chars = list(map(list, expected_first_sentence_words))
        assert feature.first_sentence_tokenized == {
            "words": expected_first_sentence_words,
            "characters": expected_first_sentence_chars
        }
        expected_second_sentence_words = ["does", "he", "enjoy", "coding",
                                          "in", "the", "rain", "?"]
        expected_second_sentence_chars = list(map(list, expected_second_sentence_words))
        assert feature.second_sentence_tokenized == {
            "words": expected_second_sentence_words,
            "characters": expected_second_sentence_chars
        }
        assert feature.second_sentence_str == question2
        assert feature.label == 0
        with self.assertRaises(RuntimeError):
            PairFeature.read_from_line("This is not a proper line.")

    def test_to_indexed_feature_converts_correctly(self):
        feature = PairFeature("What do dogs eat?",
                               "What do cats eat, play with, or enjoy?",
                               0)
        data_indexer = DataIndexer()
        what_index = data_indexer.add_word_to_index("what")
        do_index = data_indexer.add_word_to_index("do")
        dogs_index = data_indexer.add_word_to_index("dogs")
        eat_index = data_indexer.add_word_to_index("eat")
        cats_index = data_indexer.add_word_to_index("cats")
        question_index = data_indexer.add_word_to_index("?")
        comma_index = data_indexer.add_word_to_index(",")
        play_index = data_indexer.add_word_to_index("play")
        with_index = data_indexer.add_word_to_index("with")
        or_index = data_indexer.add_word_to_index("or")
        enjoy_index = data_indexer.add_word_to_index("enjoy")
        idxd_feature = feature.to_indexed_feature(data_indexer)
        first_sent_word_idxs, second_sent_word_idxs = idxd_feature.get_int_word_indices()
        assert first_sent_word_idxs == [what_index,
                                        do_index,
                                        dogs_index,
                                        eat_index,
                                        question_index]
        assert second_sent_word_idxs == [what_index,
                                         do_index,
                                         cats_index,
                                         eat_index,
                                         comma_index,
                                         play_index,
                                         with_index,
                                         comma_index,
                                         or_index,
                                         enjoy_index,
                                         question_index]
        assert idxd_feature.label == [1, 0]

    def test_words_tokenizes_the_sentence_correctly(self):
        pair_feature = PairFeature("A sentence.",
                                   "Another sentence.",
                                   0)
        expected_words = ["a", "sentence", ".",
                          "another", "sentence", "."]
        expected_characters = ['a', 's', 'e', 'n', 't', 'e', 'n', 'c', 'e',
                               '.', 'a', 'n', 'o', 't', 'h', 'e', 'r', 's', 'e',
                               'n', 't', 'e', 'n', 'c', 'e', '.']
        assert pair_feature.words() == {"words": expected_words,
                                        "characters": expected_characters}


class TestIndexedPairFeature(DuplicateTestCase):
    def setUp(self):
        super(TestIndexedPairFeature, self).setUp()
        self.feature = IndexedPairFeature([IndexedFeatureWord(1, [1, 2]),
                                            IndexedFeatureWord(2, [3, 4]),
                                            IndexedFeatureWord(3, [5]),
                                            IndexedFeatureWord(5, [1, 4, 1]),
                                            IndexedFeatureWord(4, [1, 2, 6])],
                                           [IndexedFeatureWord(1, [1, 2]),
                                            IndexedFeatureWord(8, [3, 1, 2, 1]),
                                            IndexedFeatureWord(2, [3, 4]),
                                            IndexedFeatureWord(3, [5])],
                                           [0, 1])

    def test_get_lengths(self):
        assert self.feature.get_lengths() == {"num_sentence_words": 5,
                                               'num_word_characters': 4}

    def test_pad_adds_padding_words(self):
        self.feature.pad({"num_sentence_words": 6,
                           'num_word_characters': 5})
        first_sent_word_idxs, second_sent_word_idxs = self.feature.get_int_word_indices()
        first_sent_char_idxs, second_sent_char_idxs = self.feature.get_int_char_indices()

        assert first_sent_word_idxs == [1, 2, 3, 5, 4, 0]
        assert second_sent_word_idxs == [1, 8, 2, 3, 0, 0]
        assert first_sent_char_idxs == [[1, 2, 0, 0, 0], [3, 4, 0, 0, 0],
                                        [5, 0, 0, 0, 0], [1, 4, 1, 0, 0],
                                        [1, 2, 6, 0, 0], [0, 0, 0, 0, 0]]
        assert second_sent_char_idxs == [[1, 2, 0, 0, 0], [3, 1, 2, 1, 0],
                                         [3, 4, 0, 0, 0], [5, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        assert self.feature.label == [0, 1]

    def test_pad_truncates(self):
        self.feature.pad({"num_sentence_words": 2,
                           'num_word_characters': 3})
        first_sent_word_idxs, second_sent_word_idxs = self.feature.get_int_word_indices()
        first_sent_char_idxs, second_sent_char_idxs = self.feature.get_int_char_indices()

        assert first_sent_word_idxs == [1, 2]
        assert second_sent_word_idxs == [1, 8]
        assert first_sent_char_idxs == [[1, 2, 0], [3, 4, 0]]
        assert second_sent_char_idxs == [[1, 2, 0], [3, 1, 2]]
        assert self.feature.label == [0, 1]

    def test_pad_general(self):
        self.feature.pad(self.feature.get_lengths())
        first_sent_word_idxs, second_sent_word_idxs = self.feature.get_int_word_indices()
        first_sent_char_idxs, second_sent_char_idxs = self.feature.get_int_char_indices()

        assert first_sent_word_idxs == [1, 2, 3, 5, 4]
        assert second_sent_word_idxs == [1, 8, 2, 3, 0]
        assert first_sent_char_idxs == [[1, 2, 0, 0], [3, 4, 0, 0],
                                        [5, 0, 0, 0], [1, 4, 1, 0],
                                        [1, 2, 6, 0]]
        assert second_sent_char_idxs == [[1, 2, 0, 0], [3, 1, 2, 1],
                                         [3, 4, 0, 0], [5, 0, 0, 0],
                                         [0, 0, 0, 0]]
        assert self.feature.label == [0, 1]

    def test_as_training_data_produces_correct_numpy_arrays(self):
        self.feature.pad({'num_sentence_words': 3, 'num_word_characters': 2})
        inputs, label = self.feature.as_training_data()
        assert_allclose(label[0], np.asarray([0, 1]))
        assert len(inputs) == 2
        assert_allclose(inputs[0], np.asarray([1, 2, 3]))
        assert_allclose(inputs[1], np.asarray([1, 8, 2]))

        inputs, label = self.feature.as_training_data(mode="character")
        assert_allclose(label[0], np.asarray([0, 1]))
        assert len(inputs) == 2
        assert_allclose(inputs[0], np.asarray([[1, 2], [3, 4], [5, 0]]))
        assert_allclose(inputs[1], np.asarray([[1, 2], [3, 1], [3, 4]]))

        inputs, label = self.feature.as_training_data(mode="word+character")
        assert_allclose(label[0], np.asarray([0, 1]))
        assert len(inputs) == 4
        assert_allclose(inputs[0], np.asarray([1, 2, 3]))
        assert_allclose(inputs[1], np.asarray([[1, 2], [3, 4], [5, 0]]))
        assert_allclose(inputs[2], np.asarray([1, 8, 2]))
        assert_allclose(inputs[3], np.asarray([[1, 2], [3, 1], [3, 4]]))

    def test_as_training_data_error(self):
        with self.assertRaises(ValueError):
            feature = IndexedPairFeature([IndexedFeatureWord(1, [1, 2]),
                                           IndexedFeatureWord(4, [1, 2, 6])],
                                          [IndexedFeatureWord(1, [1, 2]),
                                           IndexedFeatureWord(3, [5])],
                                          None)
            feature.as_training_data()
        with self.assertRaises(ValueError):
            self.feature.as_training_data(mode="words+character")

    def test_as_testing_data_produces_correct_numpy_arrays(self):
        self.feature.pad({'num_sentence_words': 4, 'num_word_characters': 2})
        inputs, labels = self.feature.as_testing_data()
        assert len(labels) == 0
        assert len(inputs) == 2
        assert_allclose(inputs[0], np.asarray([1, 2, 3, 5]))
        assert_allclose(inputs[1], np.asarray([1, 8, 2, 3]))

        inputs, label = self.feature.as_training_data(mode="character")
        assert len(labels) == 0
        assert len(inputs) == 2
        assert_allclose(inputs[0], np.asarray([[1, 2], [3, 4], [5, 0], [1, 4]]))
        assert_allclose(inputs[1], np.asarray([[1, 2], [3, 1], [3, 4], [5, 0]]))

        inputs, label = self.feature.as_training_data(mode="word+character")
        assert len(labels) == 0
        assert len(inputs) == 4
        assert_allclose(inputs[0], np.asarray([1, 2, 3, 5]))
        assert_allclose(inputs[1], np.asarray([[1, 2], [3, 4], [5, 0], [1, 4]]))
        assert_allclose(inputs[2], np.asarray([1, 8, 2, 3]))
        assert_allclose(inputs[3], np.asarray([[1, 2], [3, 1], [3, 4], [5, 0]]))

    def test_as_testing_data_error(self):
        with self.assertRaises(ValueError):
            self.feature.as_testing_data(mode="words+character")

    def test_equals(self):
        feature_1 = IndexedPairFeature([IndexedFeatureWord(1, [1, 2]),
                                         IndexedFeatureWord(4, [1, 2, 6])],
                                        [IndexedFeatureWord(1, [1, 2]),
                                         IndexedFeatureWord(3, [5])],
                                        None)

        feature_2 = IndexedPairFeature([IndexedFeatureWord(1, [1, 2]),
                                         IndexedFeatureWord(4, [1, 2, 6])],
                                        [IndexedFeatureWord(1, [1, 2]),
                                         IndexedFeatureWord(3, [5])],
                                        None)

        feature_3 = IndexedPairFeature([IndexedFeatureWord(1, [1, 2]),
                                         IndexedFeatureWord(1, [2, 2])],
                                        [IndexedFeatureWord(1, [1, 2]),
                                         IndexedFeatureWord(3, [5])],
                                        None)
        feature_4 = IndexedPairFeature([IndexedFeatureWord(1, [1, 2])],
                                        [IndexedFeatureWord(1, [2, 2])],
                                        None)
        self.assertNotEquals(feature_1, feature_4)
        self.assertNotEquals(feature_1, feature_3)
        self.assertFalse(feature_1.__eq__(0))
        self.assertEquals(feature_1, feature_2)

    def test_less_than(self):
        feature_1 = IndexedPairFeature([IndexedFeatureWord(1, [1, 2]),
                                         IndexedFeatureWord(4, [1, 2, 6])],
                                        [IndexedFeatureWord(1, [1, 2]),
                                         IndexedFeatureWord(3, [5])],
                                        None)

        feature_2 = IndexedPairFeature([IndexedFeatureWord(1, [1, 2]),
                                         IndexedFeatureWord(4, [1, 2, 6])],
                                        [IndexedFeatureWord(2, [2, 2]),
                                         IndexedFeatureWord(3, [5])],
                                        None)

        feature_3 = IndexedPairFeature([IndexedFeatureWord(1, [1, 2])],
                                        [IndexedFeatureWord(1, [2, 2])],
                                        None)
        self.assertFalse(feature_1.__lt__(0))
        self.assertFalse(feature_2.__lt__(feature_1))
        self.assertLess(feature_1, feature_2)
        self.assertLess(feature_3, feature_2)
