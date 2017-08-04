from collections import Counter, defaultdict
import logging

import tqdm

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class DataIndexer:
    """
    A DataIndexer maps strings to integers, allowing for strings to be mapped
    to an out-of-vocabulary token.

    DataIndexers are fit to a particular dataset, which we use to decide which
    words are in-vocabulary.
    """
    def __init__(self):
        # Typically all input words to this code are lower-cased, so we could
        # simply use "PADDING" for this. But doing it this way, with special
        # characters, future-proofs the code in case it is used later in a
        # setting where not all input is lowercase.
        self._padding_token = "@@PADDING@@"
        self._oov_token = "@@UNKOWN@@"
        self.word_indices = defaultdict(self._default_namespace_word_indices_dict)
        self.reverse_word_indices = defaultdict(self._default_namespace_reverse_word_indices_dict)
        self.is_fit = False

    def _default_namespace_word_indices_dict(self):
        return {self._padding_token: 0, self._oov_token: 1}

    def _default_namespace_reverse_word_indices_dict(self):
        return {0: self._padding_token, 1: self._oov_token}

    def fit_word_dictionary(self, text_features, min_count=1):
        """
        Given a Dataset, this method decides which words (which could be words
        or characters) are given an index, and which ones are mapped to an OOV
        token (in this case "@@UNKNOWN@@").

        This method must be called before any dataset is indexed with this
        DataIndexer. If you don't first fit the word dictionary, you'll
        basically map every token to the OOV token. We call instance.words()
        for each instance in the dataset, and then keep all words that appear
        at least min_count times.
        :param dataset: Dataset
            The dataset to index.
        :param min_count: int
            The minimum number of times a word must occur in order
            to be indexed. 
        :return: 
        """
        # if not isinstance(dataset, Dataset):
        #     raise ValueError("Expected dataset to be type "
        #                      "Dataset, found {} of type "
        #                      "{}".format(dataset, type(dataset)))
        if not isinstance(min_count, int):
            raise ValueError("Expected min_count to be type "
                             "int, found {} of type "
                             "{}".format(min_count, type(min_count)))

        logger.info("Fitting word dictionary with min count of %d", min_count)
        namespace_word_counts = defaultdict(Counter)
        for feature in tqdm.tqdm(text_features):
            # dictionary with keys as namespace names, and values as array
            # the words for that namespace.
            namespace_dict = feature.words()
            for namespace in namespace_dict:
                for word in namespace_dict[namespace]:
                    namespace_word_counts[namespace][word] += 1
        # Index the dataset, sorted by order of decreasing frequency, and then
        # alphabetically for ties.
        for namespace, word_counts in namespace_word_counts.items():
            sorted_word_counts = sorted(word_counts.items(),
                                        key=lambda pair: (-pair[1],
                                                          pair[0]))
            for word, count in sorted_word_counts:
                if count >= min_count:
                    self.add_word_to_index(word, namespace)
        self.is_fit = True

    def add_word_to_index(self, word, namespace="words"):
        """
        Adds `word` to the index, if it is not already present. Either way, we
        return the index of the word.
        
        :param word: str
            A string to be added to the indexer.

        :param namespace: str
            The string namespace to index the word under.

        :return: index: int
            The index of the input word in the namespace.
        """
        if not isinstance(word, str):
            raise ValueError("Expected word to be type "
                             "str, found {} of type "
                             "{}".format(word, type(word)))
        if word not in self.word_indices[namespace]:
            index = len(self.word_indices[namespace])
            self.word_indices[namespace][word] = index
            self.reverse_word_indices[namespace][index] = word
            return index
        else:
            return self.word_indices[namespace][word]

    def words_in_index(self, namespace="words"):
        """
        Returns a list of the words in the index for a
        given namespace.

        :param namespace: str, optional (default="words")
            The string namespace to return the list of words
            in.

        :return:word_list: List of str
            A list of the words added to this DataIndexer.
        """
        return self.word_indices[namespace].keys()

    def get_word_index(self, word, namespace="words"):
        """
        Get the index of a word.

        :param word: str
            A string to return the index of.

        :param namespace: str, optional (default="words")
            The string namespace to return the list of words
            in.

        :return: index: int
            The index of the input word if it is in the index, or the index
            corresponding to the OOV token if it is not.
        """
        if not isinstance(word, str):
            raise ValueError("Expected word to be type "
                             "str, found {} of type "
                             "{}".format(word, type(word)))
        if word in self.word_indices[namespace]:
            return self.word_indices[namespace][word]
        else:
            return self.word_indices[namespace][self._oov_token]

    def get_word_from_index(self, index, namespace="words"):
        """
        Get the word corresponding to an input index, for a
        given namespace.

        :param index: int
            The int index to retrieve the word from.

        :param namespace: str, optional (default="words")
            The string namespace to return the list of words
            in.

        :return: word: str
            The string word occupying the input index.
        """
        if not isinstance(index, int):
            raise ValueError("Expected index to be type "
                             "int, found {} of type "
                             "{}".format(index, type(index)))
        return self.reverse_word_indices[namespace][index]

    def get_vocab_size(self, namespace="words"):
        """
        Get the number of words in a namespace.

        :param namespace: str, optional (default="words")
            The string namespace to return the list of words
            in.

        :return: vocab_size: int
            The number of words added to this DataIndexer.
        """
        return len(self.word_indices[namespace])

    def get_index_to_word(self):
        return self.reverse_word_indices