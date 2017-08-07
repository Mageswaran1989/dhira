import codecs
import itertools
import logging
import os
import numpy as np

from tqdm import tqdm_notebook as tqdm

from dhira.data.download.downloader import Downloader
from dhira.data.utils.pickle_data import PickleData

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Dataset(PickleData):
    """
    A collection of Features. This base class has general methods that apply
    to all collections of Features. That basically is just methods that
    operate on sets, like merging and truncating.
    """
    def __init__(self,
                 name='default',
                 feature_type = None,
                 train_files = None,
                 test_files = None,
                 val_files = None,
                 train_features = None,
                 val_features=None,
                 test_features=None,
                 download_path=None):
        """
        Initializes the Dataset with the feature type to be read and the required files
        :param feature_type: Sub class of `Feature` type
        :param train_files: : List[str]
            A collection of filenames to read the specific self.feature_type
            from, line by line. 
        :param test_files: : List[str]
            A collection of filenames to read the specific self.feature_type
            from, line by line.
        :param val_files: : List[str]
            A collection of filenames to read the specific self.feature_type
            from, line by line.
        """
        super(Dataset, self).__init__(pickle_directory=name)
        self.download_path = download_path
        self.name = name
        self.feature_type = feature_type

        self.train_features = train_features
        self.val_features = val_features
        self.test_features = test_features
        self.train_files = train_files
        self.test_files = test_files
        self.val_files = val_files

        self.train_pickle_file = self.name + '-train.p'
        self.val_pickle_file = self.name + '-val.p'
        self.test_pickle_file = self.name +  '-test.p'
        self.indexer_pickle_file = self.name + '-data_indexer.p'


    def download(self, url: str):
        """
        Downloads the data set and extracts to the provided download_path
        """
        if self.download_path is None:
            #creates a folder @ ~/.dhira/dataset_name/
            self.download_path = os.path.expanduser(os.path.join('~', '.dhira', self.name))
            if not os.path.exists(self.download_path):
                os.makedirs(self.download_path)
        else:
            if not os.path.exists(self.download_path):
                os.makedirs(self.download_path)
        file = self.download_path + '/' + url.split('/')[-1]
        downloaded_path = Downloader.get(url, file)
        if('tar' in file): downloaded_path = Downloader.extract_tar(file, self.download_path) #TODO check file type
        return downloaded_path

    def merge(self, other):
        """
        Combine two datasets. If you call try to merge two Datasets of the same
        subtype, you will end up with a Dataset of the same type (i.e., calling
        IndexedDataset.merge() with another IndexedDataset will return an
        IndexedDataset). If the types differ, this method currently raises an
        error, because the underlying Feature objects are not currently type
        compatible.
        :param other: the Dataset that needs to be merged
        :return: Merged Dataset
        """
        if type(self) is type(other):
            return self.__class__(self.features + other.features)
        else:
            raise ValueError("Cannot merge datasets with different types")

    @staticmethod
    def truncate(features, max_features):
        """
        Truncate the dataset to a fixed size.
        :param max_features (int): 
         The maximum amount of features allowed in this dataset. If there
            are more features than `max_features` in this dataset, we
            return a new dataset with a random subset of size `max_features`.
            If there are fewer than `max_features` already, we just return
            self.
        :return: 
        """
        if not isinstance(max_features, int):
            raise ValueError("Expected max_features to be type "
                             "int, found {} of type "
                             "{}".format(max_features, type(max_features)))
        if max_features < 1:
            raise ValueError("max_features must be at least 1"
                             ", found {}".format(max_features))

        if len(features) <= max_features:
            return features

        # new_features = [i for i in self.features] TODO: Remove
        return features[:max_features]

    @classmethod
    def to_features(cls, file_names, feature_type):
        """
        Read a dataset (basically a list of features) from
        a data file.
        :param file_names: str or List of str
                 The string filename from which to read the features, or a List of
            strings repesenting the files to pull features from. If a string
            is passed in, it is automatically converted to a single-element
            list.
        :param feature_class: Feature
            The Feature class to create from these lines.
        :return: text_dataset: TextDataset
            A new TextDataset with the features read from the file.
        """

        if isinstance(file_names, str):
            file_names = [file_names]
        # If file_names is not a list, throw an error. If it is a list,
        # but the first element isn't a string, also throw an error.
        if not isinstance(file_names, list) or not isinstance(file_names[0],
                                                              str):
            raise ValueError("Expected filename to be a List of strings "
                             "but was {} of type "
                             "{}".format(file_names, type(file_names)))
        logger.info("Reading files {} to a list of lines.".format(file_names))
        lines = [x.strip() for filename in file_names
                 for x in tqdm(codecs.open(filename, "r", "utf-8").readlines())]
        return cls.lines_to_features(lines, feature_type)

    @classmethod
    def lines_to_features(cls, lines, feature_type):
        """
        Read a dataset (basically a list of features) from
        a data file.
        :param lines: List of str
            A list containing string representations of each
            line in the file.
        :param feature_class: Feature
            The Feature class to create from these lines.
        :return: text_dataset: TextDataset
            A new TextDataset with the features read from the list.
        """
        if not isinstance(lines, list):
            raise ValueError("Expected lines to be a list, "
                             "but was {} of type "
                             "{}".format(lines, type(lines)))

        if not isinstance(lines[0], str):
            raise ValueError("Expected lines to be a list of strings, "
                             "but the first element of the list was {} "
                             "of type {}".format(lines[0], type(lines[0])))

        logger.info("Creating list of {} features from "
                    "list of lines.".format(feature_type))

        features = [feature_type.read_from_line(line) for line in tqdm(lines)]

        # TODO remove below user info???
        labels = [(x.label, x) for x in features]
        labels.sort(key=lambda x: str(x[0]))
        label_counts = [(label, len([x for x in group]))
                        for label, group
                        in itertools.groupby(labels, lambda x: x[0])]
        label_count_str = str(label_counts)
        if len(label_count_str) > 100:
            label_count_str = label_count_str[:100] + '...'
        logger.info("Finished reading dataset; label counts: %s",
                    label_count_str)

        return features

    def split_dataset(self, features_list):
        """
        Divides the features list in to 8:1:1 train:val:test sets
        :param features_list: 
        :return: 
        """
        np.random.seed(42)
        train_indices = np.random.choice(len(features_list), round(len(features_list) * 0.8), replace=False)
        test_val_indices = np.array(list(set(range(len(features_list))) - set(train_indices)))
        mid_point = int(len(test_val_indices) / 2)
        val_indices = test_val_indices[:mid_point]
        test_indices = test_val_indices[mid_point:]

        self.train_features = [features_list[index] for index in train_indices]
        self.val_features = [features_list[index] for index in val_indices]
        self.test_features = [features_list[index] for index in test_indices]


    def pickle_train_features(self):
        if not self.check_pickle_exists(self.train_pickle_file):
            logger.info("Pickling the train data file")
            self.write_pickle(self.train_features, self.train_pickle_file)
        else:
            logger.info('{} already exists'.format(str(self.train_pickle_file)))

    def pickle_val_features(self):
        if not self.check_pickle_exists(self.val_pickle_file):
            logger.info("Pickling the val data file")
            self.write_pickle(self.train_features, self.val_pickle_file)
        else:
            logger.info('{} already exists'.format(str(self.val_pickle_file)))

    def pickle_test_features(self):
        if not self.check_pickle_exists(self.test_pickle_file):
            logger.info("Pickling the test data file")
            self.write_pickle(self.train_features, self.test_pickle_file)
        else:
            logger.info('{} already exists'.format(str(self.test_pickle_file)))

    def _load_train_features(self):
        raise NotImplementedError

    def _load_val_features(self):
        raise NotImplementedError

    def _load_test_features(self):
        raise NotImplementedError

    def load_train_features(self):

        logger.info("Getting training data from {}".format(self.train_files))

        if not self.check_pickle_exists(self.train_pickle_file):
            logger.info("Processing the train data file for first time")
            self._load_train_features()
        else:
            logger.info("Reusing the pickle file {}.".format(self.train_pickle_file))
            self.train_features = self.read_pickle(self.train_pickle_file)

    def load_val_features(self):
        logger.info("Getting validation data from {}".format(self.val_files))

        if not self.check_pickle_exists(self.val_pickle_file):
            logger.info("Processing the validation data file for first time")
            self._load_val_features()
        else:
            logger.info("Reusing the pickle file {}.".format(self.val_features))
            self.val_features = self.read_pickle(self.val_pickle_file)

    def load_test_features(self):
        logger.info("Getting test data from {}".format(self.test_files))

        if not self.check_pickle_exists(self.test_pickle_file):
            logger.info("Processing the test data file for first time")
            self._load_test_features()
        else:
            logger.info("Reusing the pickle file {}.".format(self.test_features))
            self.test_features = self.read_pickle(self.test_pickle_file)

    #Override this for more control at feature level

    # This is a hack to get the function to run the code above immediately,
    # instead of doing the standard python generator lazy-ish evaluation.
    # This is necessary to set the class variables ASAP.
    def get_train_batch_generator(self):
        for feature in self.train_features:
            # Now, we want to take the instance and convert it into
            # NumPy arrays suitable for training.
            inputs, labels = feature.as_training_data()
            yield inputs, labels

    def get_validation_batch_generator(self):
        for feature in self.val_features:
            # Now, we want to take the instance and convert it into
            # NumPy arrays suitable for training.
            inputs, labels = feature.as_validation_data()
            yield inputs, labels

    def get_test_batch_generator(self):
        for feature in self.test_features:
            # Now, we want to take the instance and convert it into
            # NumPy arrays suitable for training.
            inputs, labels = feature.as_testing_data()
            yield inputs, labels

#-------------------------------------------------------------------------------------
