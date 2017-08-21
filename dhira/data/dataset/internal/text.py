import logging

from tqdm import tqdm_notebook as tqdm

from dhira.data.data_indexer import DataIndexer
from dhira.data.dataset.internal.dataset_base import Dataset
from dhira.data.features.indexed_feature import IndexedFeature
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class TextDataset(Dataset):
    """
    A Dataset of TextFeatures, with a few helper methods. TextFeatures aren't
    useful for much until they've been indexed. So this class just has methods
    to read in data from a file and converting it into other kinds of Datasets.
    """

    def __init__(self,
                 name='default',
                 feature_type=None,
                 train_files=None,
                 test_files=None,
                 val_files=None,
                 train_features=None,
                 val_features=None,
                 test_features=None,
                 download_path=None,
                 min_count = 1,
                 pad=True,
                 max_lengths=30,
                 index_mode="word"):
        """
        Construct a new TextDataset
        :param features: List of TextFeature
                A list of TextFeatures to construct
                    the TextDataset from.
        """
        super(TextDataset, self).__init__(name=name,
                                          feature_type=feature_type,
                                          train_files=train_files,
                                          test_files = test_files,
                                          val_files=val_files,
                                          train_features=train_features,
                                          val_features=val_features,
                                          test_features=test_features,
                                          download_path=download_path)

        self.data_indexer: DataIndexer = DataIndexer()
        self.min_count = min_count
        self.is_data_indexer_fitted = False
        self._is_train_data_indexed = False
        self._is_val_data_indexed = False
        self._is_test_data_indexed = False
        self.pad = pad
        self.max_lengths = max_lengths
        self.index_mode = index_mode
        self.nlp = None

        # We now need to check if the user specified max_length for
        # the feature, and accordingly truncate or pad if applicable. If
        # max_length is None for a given string key, we assume that no
        # truncation is to be done and the max lengths should be read from the
        # features.
        if not self.pad and self.max_lengths:
            raise ValueError("Passed in max_length {}, but set pad to false. "
                             "Did you mean to do this?".format(self.max_lengths))

        #Load the DataIndexer if found
        if self.check_pickle_exists(self.indexer_pickle_file):
            logger.info("Reusing the pickle file {}.".format(self.indexer_pickle_file))
            self.data_indexer = self.read_pickle(self.indexer_pickle_file)
            self.is_data_indexer_fitted = True

            #Lets safely assume data is already fitted in previous oass
            self._is_train_data_indexed = True
            self._is_val_data_indexed = True
            self._is_test_data_indexed = True

    def set_nlp_pipelie(self, nlp):
        """
        Provision for DataManager to load spaCy nlp pipeline from user environment 
        :param nlp: 
        :return: 
        """
        self.nlp = nlp

    def get_max_lengths(self, features):
        """

        :return: 
        """
        max_lengths = {}
        lengths = [feature.get_lengths() for feature in features]
        if not lengths:
            return max_lengths
        for key in lengths[0]:
            max_lengths[key] = max(x[key] if key in x else 0 for x in lengths)
        return max_lengths

    def max_lengths_to_use(self, features):
        # Get max lengths from the dataset
        dataset_max_lengths = self.get_max_lengths(features)
        logger.info("Instance max lengths {}".format(dataset_max_lengths))
        max_lengths_to_use = dataset_max_lengths
        if self.pad:
            # If the user set max lengths, iterate over the
            # dictionary provided and verify that they did not
            # pass any keys to truncate that are not in the feature.
            if self.max_lengths is not None:
                for input_dimension, length in self.max_lengths.items():
                    if input_dimension in dataset_max_lengths:
                        max_lengths_to_use[input_dimension] = length
                    else:
                        raise ValueError("Passed a value for the max_lengths "
                                         "that does not exist in the "
                                         "feature. Improper input length "
                                         "dimension (key) we found was {}, "
                                         "lengths dimensions in the feature "
                                         "are {}".format(input_dimension,
                                                         dataset_max_lengths.keys()))
            logger.info("Padding lengths to length: {}".format(str(max_lengths_to_use)))
        return max_lengths_to_use

    def pad_features(self, features, max_lengths=None):
        """
        Make all of the IndexedFeatures in the dataset have the same length
        by padding them (in the front) with zeros.
        If max_length is given for a particular dimension, we will pad all
        features to that length (including left-truncating features if
        necessary). If not, we will find the longest feature and pad all
        features to that length. Note that max_lengths is a _List_, not an int
        - there could be several dimensions on which we need to pad, depending
        on what kind of feature we are dealing with.
        This method _modifies_ the current object, it does not return a new
        IndexedDataset.
        """
        # First we need to decide _how much_ to pad. To do that, we find the
        # max length for all relevant padding decisions from the features
        # themselves. Then we check whether we were given a max length for a
        # particular dimension. If we were, we use that instead of the
        # feature-based one.
        logger.info("Getting max lengths from features")
        feature_max_lengths = self.get_max_lengths(features)
        logger.info("Feature max lengths: %s", str(feature_max_lengths))
        lengths_to_use = {}
        for key in feature_max_lengths:
            if max_lengths and max_lengths[key] is not None:
                lengths_to_use[key] = max_lengths[key]
            else:
                lengths_to_use[key] = feature_max_lengths[key]

        logger.info("Now actually padding features to length: %s",
                    str(lengths_to_use))
        for feature in tqdm(features):
            feature.pad(lengths_to_use)

        return features

    def fit_data_indexer(self):
        logger.info('Trying to load prefitted data_indexer... {}'.format(self.indexer_pickle_file))
        if not self.check_pickle_exists(self.indexer_pickle_file):
            logger.info("Data Indexer needs to be fitted...")
            self.data_indexer.fit_word_dictionary(self.train_features, self.min_count)
            self.is_data_indexer_fitted = True
            logger.info("Pickling the data indexer...")
            self.write_pickle(self.data_indexer, self.indexer_pickle_file)


    def to_indexed_features(self, text_features):
        """
        Converts the text features into an indexed feature, given a DataIndexer.
        :param data_indexer: DataIndexer
                        The DataIndexer to use in converting words to indices.
        :return: Indexed Features
        """
        logger.info('Converting to indexed features...')
        indexed_features = [feature.to_indexed_feature(self.data_indexer)
                            for feature in tqdm(text_features)]
        return indexed_features

    def index_train_features(self):
        if not isinstance(self.train_features[0], IndexedFeature):
            self.train_features = self.to_indexed_features(self.train_features)
            self._is_train_data_indexed = True

        self.training_data_max_lengths = self.max_lengths_to_use(self.train_features)

    def index_val_features(self):
        if not isinstance(self.val_features[0], IndexedFeature):
            self.val_features = self.to_indexed_features(self.val_features)
            self._is_val_data_indexed = True
        self.max_lengths_to_use(self.val_features)

    def index_test_features(self):
        if not isinstance(self.test_features[0], IndexedFeature):
            self.test_features = self.to_indexed_features(self.test_features)
            self._is_test_data_indexed = True
        self.max_lengths_to_use(self.test_features)

