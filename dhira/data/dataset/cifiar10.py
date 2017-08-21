import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelBinarizer

from dhira.data.dataset.internal.dataset_base import Dataset
from dhira.data.features.image_feature import ImageFeature

logger = logging.getLogger(__name__)

class Cifiar10(Dataset):

    image_shape = (32,32,3)

    def __init__(self,
                 name='cifiar10',
                 feature_type = ImageFeature,
                 download_path=None):

        super(Cifiar10, self).__init__(name=name,
                                           feature_type=feature_type,
                                           download_path=download_path)

        self._url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        self._num_data_batches = 5 #Downloaded dataset batches
        self._downloaded_path = Dataset.download(self._url, name)
        self.preprocess_and_save_data()
        # self.test_files()

    def _load_label_names(self):
        """
        Load the label names from file
        """
        return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def load_cfar10_batch(self, batch_id):
        """
        Load a batch of the dataset
        """
        with open(self._downloaded_path + '/data_batch_' + str(batch_id), mode='rb') as file:
            batch = pickle.load(file, encoding='latin1')

        features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
        labels = batch['labels']

        return features, labels

    def normalize(self, x):
        """
        Normalize a list of sample image data in the range of 0 to 1
        : x: List of image data.  The image shape is (32, 32, 3)
        : return: Numpy array of normalize data
        """
        minV = np.min(x)
        maxV = np.max(x)
        ret = (x - minV) / maxV
        return ret

    def one_hot_encode(self, x):
        """
        One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
        : x: List of sample Labels
        : return: Numpy array of one-hot encoded labels
        """
        x_shape = len(x)
        a = np.zeros([x_shape, 10])
        for i, label in enumerate(x):
            np.put(a[i], label, 1)
        return np.array(a).reshape([x_shape, 10])

    def _preprocess_and_save(self, features, labels, filename):
        """
        Preprocess data and save it to file
        """
        features = self.normalize(features)
        labels = self.one_hot_encode(labels)

        pickle.dump((features, labels), open(filename, 'wb')) #TODO use PickleData class?

    def preprocess_and_save_data(self):
        """
        Preprocess Training and Validation Data
        """
        n_batches = self._num_data_batches
        valid_features = []
        valid_labels = []

        if os.path.exists(self._downloaded_path+'/'+'preprocess_validation.p'):
            return #TODO does this ok?

        for batch_i in range(1, n_batches + 1):
            features, labels = self.load_cfar10_batch(batch_i)
            validation_count = int(len(features) * 0.1)

            # Prprocess and save a batch of training data
            self._preprocess_and_save(
                features[:-validation_count],
                labels[:-validation_count],
                self._downloaded_path+'/'+'preprocess_batch_' + str(batch_i) + '.p')

            # Use a portion of training batch for validation
            valid_features.extend(features[-validation_count:])
            valid_labels.extend(labels[-validation_count:])

        # Preprocess and Save all validation data
        self._preprocess_and_save(
            np.array(valid_features),
            np.array(valid_labels),
            self._downloaded_path+'/'+'preprocess_validation.p')

        with open(self._downloaded_path + '/test_batch', mode='rb') as file:
            batch = pickle.load(file, encoding='latin1')

        # load the training data
        test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
        test_labels = batch['labels']

        # Preprocess and Save all training data
        self._preprocess_and_save(
            np.array(test_features),
            np.array(test_labels),
            self._downloaded_path + '/' +'preprocess_testing.p')

    def test_files(self):
        print('Testing')
        assert self._downloaded_path is not None, \
            'Cifar-10 data folder not set.'
        assert self._downloaded_path[-1] != '/', \
            'The "/" shouldn\'t be added to the end of the path.'
        assert os.path.exists(self._downloaded_path), \
            'Path not found.'
        assert os.path.isdir(self._downloaded_path), \
            '{} is not a folder.'.format(os.path.basename(self._downloaded_path))

        train_files = [self._downloaded_path + '/data_batch_' + str(batch_id) for batch_id in range(1, 6)]
        other_files = [self._downloaded_path + '/batches.meta', self._downloaded_path + '/test_batch']
        missing_files = [path for path in train_files + other_files if not os.path.exists(path)]

        assert not missing_files, \
            'Missing files in directory: {}'.format(missing_files)

        print('All files found!')

    def display_stats(self, batch_id: int, sample_id: int):
        """
        Display Stats of the the dataset
        """
        batch_ids = list(range(1, 6))

        if batch_id not in batch_ids:
            print('Batch Id out of Range. Possible Batch Ids: {}'.format(batch_ids))
            return None

        features, labels = self.load_cfar10_batch(batch_id)

        if not (0 <= sample_id < len(features)):
            print('{} samples in batch {}.  {} is out of range.'.format(len(features), batch_id, sample_id))
            return None

        print('\nStats of batch {}:'.format(batch_id))
        print('Samples: {}'.format(len(features)))
        print('Label Counts: {}'.format(dict(zip(*np.unique(labels, return_counts=True)))))
        print('First 20 Labels: {}'.format(labels[:20]))

        sample_image = features[sample_id]
        sample_label = labels[sample_id]
        label_names = self._load_label_names()

        print('\nExample of Image {}:'.format(sample_id))
        print('Image - Min Value: {} Max Value: {}'.format(sample_image.min(), sample_image.max()))
        print('Image - Shape: {}'.format(sample_image.shape))
        print('Label - Label Id: {} Name: {}'.format(sample_label, label_names[sample_label]))
        plt.axis('off')
        plt.imshow(sample_image)

    def display_image_predictions(self, features, labels, predictions):
        n_classes = 10
        label_names = self._load_label_names()
        label_binarizer = LabelBinarizer()
        label_binarizer.fit(range(n_classes))
        label_ids = label_binarizer.inverse_transform(np.array(labels))

        fig, axies = plt.subplots(nrows=4, ncols=2)
        fig.tight_layout()
        fig.suptitle('Softmax Predictions', fontsize=20, y=1.1)

        n_predictions = 3
        margin = 0.05
        ind = np.arange(n_predictions)
        width = (1. - 2. * margin) / n_predictions

        for image_i, (feature, label_id, pred_indicies, pred_values) in enumerate(
                zip(features, label_ids, predictions.indices, predictions.values)):
            pred_names = [label_names[pred_i] for pred_i in pred_indicies]
            correct_name = label_names[label_id]

            axies[image_i][0].imshow(feature)
            axies[image_i][0].set_title(correct_name)
            axies[image_i][0].set_axis_off()

            axies[image_i][1].barh(ind + margin, pred_values[::-1], width)
            axies[image_i][1].set_yticks(ind + margin)
            axies[image_i][1].set_yticklabels(pred_names[::-1])
            axies[image_i][1].set_xticks([0, 0.5, 1.0])


    def batch_features_labels(self, features, labels, batch_size):
        """
        Split features and labels into batches
        """
        for start in range(0, len(features), batch_size):
            end = min(start + batch_size, len(features))
            yield features[start:end], labels[start:end]

    def load_preprocess_training_batch(self, batch_id):
        """
        Load the Preprocessed Training data and return them in batches of <batch_size> or less
        """
        features_class = []
        filename = self._downloaded_path + '/preprocess_batch_' + str(batch_id) + '.p'
        features, labels = pickle.load(open(filename, mode='rb'))

        # Return the training data in batches of size <batch_size> or less
        # return self.batch_features_labels(features, labels, batch_size)
        for image, label in zip(features, labels):
            features_class.append(self.feature_type(image=image, label=label))
        return features_class

    def load_train_features(self):
        self.train_features = []
        for i in range(0, self._num_data_batches):
            self.train_features.extend(self.load_preprocess_training_batch(i+1))

    def load_val_features(self):
        valid_features, valid_labels = pickle.load(open(self._downloaded_path+'/preprocess_validation.p', mode='rb'))

        self.val_features = []

        for image, label in zip(valid_features, valid_labels):
            self.val_features.append(self.feature_type(image=image, label=label))

    def load_test_features(self):
        test_features, test_labels = pickle.load(open(self._downloaded_path+'/preprocess_testing.p', mode='rb'))
        #TODO rename the file
        self.test_features = []
        for image, label in zip(test_features, test_labels):
            self.test_features.append(self.feature_type(image=image, label=label))


