from overrides import overrides
from numpy.testing import assert_allclose
import numpy as np

from dhira.data.data_indexer import DataIndexer
from dhira.data.dataset import Dataset
from dhira.data.dataset import IndexedDataset
from dhira.data.dataset import TextDataset
from dhira.data.features.feature_word import IndexedFeatureWord
from dhira.data.features.pair_feature import IndexedPairFeature
from dhira.data.features.pair_feature import PairFeature

from dhira.tests.common.test_case import DuplicateTestCase


class TestDataset(DuplicateTestCase):
    def test_truncate(self):
        features = [PairFeature("testing1", "test1", None),
                     PairFeature("testing2", "test2", None)]
        dataset = Dataset(features)
        truncated = dataset.truncate(1)
        assert len(truncated.features) == 1
        with self.assertRaises(ValueError):
            truncated = dataset.truncate("1")
        with self.assertRaises(ValueError):
            truncated = dataset.truncate(0)

    def test_merge(self):
        features = [PairFeature("testing1", "test1", None),
                     PairFeature("testing2", "test2", None)]
        dataset1 = Dataset(features[:1])
        dataset2 = Dataset(features[1:])
        merged = dataset1.merge(dataset2)
        assert merged.features == features
        with self.assertRaises(ValueError):
            merged = dataset1.merge(features)

    def test_exceptions(self):
        feature = PairFeature("testing1", "test1", 0)
        with self.assertRaises(ValueError):
            Dataset(feature)
        with self.assertRaises(ValueError):
            Dataset(["not an feature"])


class TestTextDataset(DuplicateTestCase):
    def test_read_from_train_file(self):
        self.write_duplicate_questions_train_file()
        dataset = TextDataset.read_from_file(self.TRAIN_FILE, PairFeature)
        assert len(dataset.features) == 3
        feature = dataset.features[0]
        assert feature.first_sentence_str == "question1"
        assert feature.second_sentence_str == "question2 question3pad"
        assert feature.label == 0
        feature = dataset.features[1]
        assert feature.first_sentence_str == "question4"
        assert feature.second_sentence_str == "question5"
        assert feature.label == 1
        feature = dataset.features[2]
        assert feature.first_sentence_str == "question6"
        assert feature.second_sentence_str == "question7"
        assert feature.label == 0
        with self.assertRaises(ValueError):
            TextDataset.read_from_file(3, PairFeature)
        with self.assertRaises(ValueError):
            TextDataset.read_from_file([3], PairFeature)

    def test_read_from_lines(self):
        self.write_duplicate_questions_train_file()
        lines = ["\"1\",\"2\",\"3\",\"question1\",\"question2\",\"0\"\n",
                 "\"4\",\"5\",\"6\",\"question3\",\"question4\",\"1\"\n",
                 "\"7\",\"8\",\"9\",\"question5\",\"question6\",\"0\"\n"]
        dataset = TextDataset.read_from_lines(lines, PairFeature)
        assert len(dataset.features) == 3
        feature = dataset.features[0]
        assert feature.first_sentence_str == "question1"
        assert feature.second_sentence_str == "question2"
        assert feature.label == 0
        feature = dataset.features[1]
        assert feature.first_sentence_str == "question3"
        assert feature.second_sentence_str == "question4"
        assert feature.label == 1
        feature = dataset.features[2]
        assert feature.first_sentence_str == "question5"
        assert feature.second_sentence_str == "question6"
        assert feature.label == 0
        with self.assertRaises(ValueError):
            TextDataset.read_from_lines("some line", PairFeature)
        with self.assertRaises(ValueError):
            TextDataset.read_from_lines([3], "PairFeature")

    def test_read_from_test_file(self):
        self.write_duplicate_questions_test_file()
        dataset = TextDataset.read_from_file(self.TEST_FILE, PairFeature)
        assert len(dataset.features) == 3
        feature = dataset.features[0]
        assert feature.first_sentence_str == "question1 questionunk1 question1"
        assert feature.second_sentence_str == "questionunk2"
        assert feature.label is None
        feature = dataset.features[1]
        assert feature.first_sentence_str == "question3pad"
        assert feature.second_sentence_str == "question4 questionunk3"
        assert feature.label is None
        feature = dataset.features[2]
        assert feature.first_sentence_str == "question5"
        assert feature.second_sentence_str == "question6"
        assert feature.label is None
        with self.assertRaises(ValueError):
            TextDataset.read_from_file(3, PairFeature)

    def test_to_indexed_dataset(self):
        features = [PairFeature("testing1 test1", "test1", None),
                     PairFeature("testing2", "test2 testing1", None)]
        data_indexer = DataIndexer()
        testing1_index = data_indexer.add_word_to_index("testing1")
        test1_index = data_indexer.add_word_to_index("test1")
        testing2_index = data_indexer.add_word_to_index("testing2")
        test2_index = data_indexer.add_word_to_index("test2")
        dataset = TextDataset(features)
        indexed_dataset = dataset.to_indexed_dataset(data_indexer)

        indexed_feature = indexed_dataset.features[0]
        first_sent_idxs, second_sent_idxs = indexed_feature.get_int_word_indices()
        assert first_sent_idxs == [testing1_index,
                                   test1_index]
        assert second_sent_idxs == [test1_index]

        indexed_feature = indexed_dataset.features[1]
        first_sent_idxs, second_sent_idxs = indexed_feature.get_int_word_indices()
        assert first_sent_idxs == [testing2_index]
        assert second_sent_idxs == [test2_index,
                                    testing1_index]


class TestIndexedDataset(DuplicateTestCase):
    @overrides
    def setUp(self):
        super(TestIndexedDataset, self).setUp()
        self.features = [IndexedPairFeature([IndexedFeatureWord(1, [1, 5]),
                                              IndexedFeatureWord(2, [2, 1]),
                                              IndexedFeatureWord(3, [1, 4, 1])],
                                             [IndexedFeatureWord(2, [2, 1]),
                                              IndexedFeatureWord(3, [1, 4, 1])],
                                             [0, 1]),
                          IndexedPairFeature([IndexedFeatureWord(3, [1, 4, 1]),
                                              IndexedFeatureWord(1, [1, 5])],
                                             [IndexedFeatureWord(3, [1, 4, 1]),
                                              IndexedFeatureWord(1, [1, 5]),
                                              IndexedFeatureWord(3, [1, 4, 1]),
                                              IndexedFeatureWord(2, [2, 1])],
                                             [1, 0])]
        self.indexed_dataset = IndexedDataset(self.features)

    def test_max_lengths(self):
        max_lengths = self.indexed_dataset.max_lengths()
        assert max_lengths == {"num_sentence_words": 4, "num_word_characters": 3}

    def test_pad_adds_zeroes(self):
        self.indexed_dataset.pad_features({"num_sentence_words": 4,
                                            "num_word_characters": 3})
        feature = self.indexed_dataset.features[0]
        first_sent_word_idxs, second_sent_word_idxs = feature.get_int_word_indices()
        first_sent_char_idxs, second_sent_char_idxs = feature.get_int_char_indices()
        assert first_sent_word_idxs == [1, 2, 3, 0]
        assert second_sent_word_idxs == [2, 3, 0, 0]
        assert first_sent_char_idxs == [[1, 5, 0], [2, 1, 0],
                                        [1, 4, 1], [0, 0, 0]]
        assert second_sent_char_idxs == [[2, 1, 0], [1, 4, 1],
                                         [0, 0, 0], [0, 0, 0]]
        assert feature.label == [0, 1]

        feature = self.indexed_dataset.features[1]
        first_sent_word_idxs, second_sent_word_idxs = feature.get_int_word_indices()
        first_sent_char_idxs, second_sent_char_idxs = feature.get_int_char_indices()
        assert first_sent_word_idxs == [3, 1, 0, 0]
        assert second_sent_word_idxs == [3, 1, 3, 2]
        assert first_sent_char_idxs == [[1, 4, 1], [1, 5, 0],
                                        [0, 0, 0], [0, 0, 0]]
        assert second_sent_char_idxs == [[1, 4, 1], [1, 5, 0],
                                         [1, 4, 1], [2, 1, 0]]
        assert feature.label == [1, 0]

    def test_pad_truncates(self):
        self.indexed_dataset.pad_features({"num_sentence_words": 2,
                                            "num_word_characters": 1})
        feature = self.indexed_dataset.features[0]
        first_sent_word_idxs, second_sent_word_idxs = feature.get_int_word_indices()
        first_sent_char_idxs, second_sent_char_idxs = feature.get_int_char_indices()
        assert first_sent_word_idxs == [1, 2]
        assert second_sent_word_idxs == [2, 3]
        assert first_sent_char_idxs == [[1], [2]]
        assert second_sent_char_idxs == [[2], [1]]
        assert feature.label == [0, 1]

        feature = self.indexed_dataset.features[1]
        first_sent_word_idxs, second_sent_word_idxs = feature.get_int_word_indices()
        first_sent_char_idxs, second_sent_char_idxs = feature.get_int_char_indices()
        assert first_sent_word_idxs == [3, 1]
        assert second_sent_word_idxs == [3, 1]
        assert first_sent_char_idxs == [[1], [1]]
        assert second_sent_char_idxs == [[1], [1]]
        assert feature.label == [1, 0]

    def test_as_training_data(self):
        self.indexed_dataset.pad_features(self.indexed_dataset.max_lengths())
        inputs, labels = self.indexed_dataset.as_training_data()

        first_sentence, second_sentence = inputs[0]
        label = labels[0]
        assert_allclose(first_sentence, np.array([1, 2, 3, 0]))
        assert_allclose(second_sentence, np.array([2, 3, 0, 0]))
        assert_allclose(label[0], np.array([0, 1]))

        first_sentence, second_sentence = inputs[1]
        label = labels[1]
        assert_allclose(first_sentence, np.array([3, 1, 0, 0]))
        assert_allclose(second_sentence, np.array([3, 1, 3, 2]))
        assert_allclose(label[0], np.array([1, 0]))

        inputs, labels = self.indexed_dataset.as_training_data(mode="character")
        first_sentence, second_sentence = inputs[0]
        label = labels[0]
        assert_allclose(first_sentence, np.array([[1, 5, 0], [2, 1, 0],
                                                  [1, 4, 1], [0, 0, 0]]))
        assert_allclose(second_sentence, np.array([[2, 1, 0], [1, 4, 1],
                                                   [0, 0, 0], [0, 0, 0]]))
        assert_allclose(label[0], np.array([0, 1]))

        first_sentence, second_sentence = inputs[1]
        label = labels[1]
        assert_allclose(first_sentence, np.array([[1, 4, 1], [1, 5, 0],
                                                  [0, 0, 0], [0, 0, 0]]))
        assert_allclose(second_sentence, np.array([[1, 4, 1], [1, 5, 0],
                                                   [1, 4, 1], [2, 1, 0]]))

        inputs, labels = self.indexed_dataset.as_training_data(mode="word+character")
        (first_sentence_words, first_sentence_characters,
         second_sentence_words, second_sentence_characters) = inputs[0]
        label = labels[0]
        assert_allclose(first_sentence_words, np.array([1, 2, 3, 0]))
        assert_allclose(second_sentence_words, np.array([2, 3, 0, 0]))
        assert_allclose(first_sentence_characters, np.array([[1, 5, 0], [2, 1, 0],
                                                             [1, 4, 1], [0, 0, 0]]))
        assert_allclose(second_sentence_characters, np.array([[2, 1, 0], [1, 4, 1],
                                                              [0, 0, 0], [0, 0, 0]]))
        assert_allclose(label[0], np.array([0, 1]))

        (first_sentence_words, first_sentence_characters,
         second_sentence_words, second_sentence_characters) = inputs[1]
        label = labels[1]
        assert_allclose(first_sentence_words, np.array([3, 1, 0, 0]))
        assert_allclose(second_sentence_words, np.array([3, 1, 3, 2]))
        assert_allclose(first_sentence_characters, np.array([[1, 4, 1], [1, 5, 0],
                                                             [0, 0, 0], [0, 0, 0]]))
        assert_allclose(second_sentence_characters, np.array([[1, 4, 1], [1, 5, 0],
                                                              [1, 4, 1], [2, 1, 0]]))

    def test_as_testing_data(self):
        features = [IndexedPairFeature([IndexedFeatureWord(1, [1, 4, 4]),
                                         IndexedFeatureWord(2, [2, 3]),
                                         IndexedFeatureWord(3, [5, 1])],
                                        [IndexedFeatureWord(2, [2, 3]),
                                         IndexedFeatureWord(3, [5, 1])],
                                        None),
                     IndexedPairFeature([IndexedFeatureWord(3, [5, 1]),
                                         IndexedFeatureWord(1, [1, 4, 4])],
                                        [IndexedFeatureWord(3, [5, 1]),
                                         IndexedFeatureWord(1, [1, 4, 4]),
                                         IndexedFeatureWord(3, [5, 1]),
                                         IndexedFeatureWord(2, [2, 3])],
                                        None)]
        indexed_dataset = IndexedDataset(features)
        indexed_dataset.pad_features(indexed_dataset.max_lengths())
        inputs, labels = indexed_dataset.as_testing_data()
        assert len(labels) == 0

        first_sentence, second_sentence = inputs[0]
        assert_allclose(first_sentence, np.array([1, 2, 3, 0]))
        assert_allclose(second_sentence, np.array([2, 3, 0, 0]))

        first_sentence, second_sentence = inputs[1]
        assert_allclose(first_sentence, np.array([3, 1, 0, 0]))
        assert_allclose(second_sentence, np.array([3, 1, 3, 2]))

        inputs, labels = indexed_dataset.as_testing_data(mode="character")
        assert len(labels) == 0

        first_sentence, second_sentence = inputs[0]
        assert_allclose(first_sentence, np.array([[1, 4, 4], [2, 3, 0],
                                                  [5, 1, 0], [0, 0, 0]]))
        assert_allclose(second_sentence, np.array([[2, 3, 0], [5, 1, 0],
                                                   [0, 0, 0], [0, 0, 0]]))

        first_sentence, second_sentence = inputs[1]
        assert_allclose(first_sentence, np.array([[5, 1, 0], [1, 4, 4],
                                                  [0, 0, 0], [0, 0, 0]]))
        assert_allclose(second_sentence, np.array([[5, 1, 0], [1, 4, 4],
                                                   [5, 1, 0], [2, 3, 0]]))

        inputs, labels = indexed_dataset.as_testing_data(mode="word+character")
        assert len(labels) == 0

        (first_sentence_words, first_sentence_characters,
         second_sentence_words, second_sentence_characters) = inputs[0]
        assert_allclose(first_sentence_words, np.array([1, 2, 3, 0]))
        assert_allclose(second_sentence_words, np.array([2, 3, 0, 0]))
        assert_allclose(first_sentence_characters, np.array([[1, 4, 4], [2, 3, 0],
                                                             [5, 1, 0], [0, 0, 0]]))
        assert_allclose(second_sentence_characters, np.array([[2, 3, 0], [5, 1, 0],
                                                              [0, 0, 0], [0, 0, 0]]))

        (first_sentence_words, first_sentence_characters,
         second_sentence_words, second_sentence_characters) = inputs[1]
        assert_allclose(first_sentence_words, np.array([3, 1, 0, 0]))
        assert_allclose(second_sentence_words, np.array([3, 1, 3, 2]))
        assert_allclose(first_sentence_characters, np.array([[5, 1, 0], [1, 4, 4],
                                                             [0, 0, 0], [0, 0, 0]]))
        assert_allclose(second_sentence_characters, np.array([[5, 1, 0], [1, 4, 4],
                                                              [5, 1, 0], [2, 3, 0]]))
        with self.assertRaises(ValueError):
            indexed_dataset.as_testing_data(mode="char")

    def test_sort(self):
        # lengths: 3, 4, 1, 2, 2
        sorted_features = [IndexedPairFeature([IndexedFeatureWord(3, [1, 4, 1]),
                                                IndexedFeatureWord(1, [1, 5])],
                                               [IndexedFeatureWord(3, [1, 4, 1]),
                                                IndexedFeatureWord(1, [1, 5]),
                                                IndexedFeatureWord(3, [1, 4, 1]),
                                                IndexedFeatureWord(2, [2, 1])],
                                               [1, 0]),
                            IndexedPairFeature([IndexedFeatureWord(1, [1, 5]),
                                                IndexedFeatureWord(2, [2, 1]),
                                                IndexedFeatureWord(3, [1, 4, 1])],
                                               [IndexedFeatureWord(2, [2, 1]),
                                                IndexedFeatureWord(3, [1, 4, 1])],
                                               [0, 1])]
        self.assertNotEqual(sorted_features, self.indexed_dataset.features)
        self.indexed_dataset.sort()
        self.assertEquals(sorted_features, self.indexed_dataset.features)
