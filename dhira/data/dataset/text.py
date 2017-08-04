import codecs
import itertools
import logging
import sys
from tqdm import tqdm_notebook as tqdm

from dhira.data.dataset.dataset_base import Dataset

from dhira.data.data_indexer import DataIndexer

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
                 pickle_dir=None):
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
                                          pickle_dir=pickle_dir)

    @staticmethod
    def fit_data_indexer(text_features, min_count: int):
        data_indexer: DataIndexer = DataIndexer()
        data_indexer.fit_word_dictionary(text_features, min_count)
        return data_indexer

    @staticmethod
    def to_indexed_features(text_features, data_indexer):
        """
        Converts the Dataset into an IndexedDataset, given a DataIndexer.
        :param data_indexer: DataIndexer
                        The DataIndexer to use in converting words to indices.
        :return: IndexedDataset
        """
        logger.info('Converting to indexed features...')
        indexed_features = [feature.to_indexed_feature(data_indexer)
                            for feature in tqdm(text_features)]
        return indexed_features
