import pandas as pd
from keras.preprocessing import sequence
import numpy as np
from tqdm import tqdm
import pickle

from dhira.embeddings_loader import EmbeddingLoader
from dhira.global_config import Config, NLPEngine
from dhira.sentence_tokenizer import *

def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

class InputProcessor:

    def __init__(self):
        np.random.seed(131)

        self.PICKLE_TRAIN_DATA_SET_FILE = "train_data_set.p"
        self.PICKLE_VALIDATION_DATA_SET_FILE = "validation_data_set.p"
        self.PICKLE_TEST_DATA_SET_FILE = "test_data_set.p"

        self.train_data = None
        self.test_data = None
        self.tk = None
        self.word_index = None

        self.train_x1 = None
        self.train_x2 = None
        self.train_y = None

        self.test_x1 = None
        self.test_x2 = None
        self.test_id = None

        self.embedding_loader = None

    def init_tokenizer(self):
        self.tk = Tokenizer()

    def pickle_data_set(self):
        raise NotImplementedError("Needs to be implemented for specific Dataset")

    @property
    def get_word_index(self):
        return self.tk.word_index

    def load_data(self):
        raise NotImplementedError("Needs to be implemented for specific Dataset")

    def fit_tokenizer(self):
        raise NotImplementedError("Needs to be implemented for specific Dataset")

    def process_with_tokenizer(self):
        raise NotImplementedError("Needs to be implemented for specific Dataset")

    def preprocess(self):
        print("Reading the test and train data...")
        self.load_data()
        self.init_tokenizer()
        print("Lets train the Tokenizer for our corpus...")
        print('Fitting Tokenizer')
        self.fit_tokenizer()
        print('Processing the data set')
        self.process_with_tokenizer()
        print('Pickling the data set')
        self.pickle_data_set()
        print('Pickling Embedding Matrix...')
        self.embedding_loader = EmbeddingLoader.get(config=NLPEngine.NONE) # TODO rewire the loader
        self.embedding_loader.pickle_embeddings()

##########################################################################################

class QuoraInputProcessor(InputProcessor):

    def __init__(self):
        InputProcessor.__init__(self)

    def load_data(self):
        self.train_data = pd.read_csv(Config.TRAIN_DATA_PATH)
        self.test_data = pd.read_csv(Config.TEST_DATA_PATH)

    def fit_tokenizer(self):
        #TODO use column names to get the data, thus making this more generic
        self.train_x1 = self.tk.text_to_seq(list(self.train_data.question1.values.astype(str)))
        self.train_x2 = self.tk.text_to_seq(list(self.train_data.question2.values.astype(str)))
        self.train_y = self.train_data.is_duplicate.values.astype(int)

        self.test_x1 = self.tk.text_to_seq(list(self.test_data.question1.values.astype(str)))
        self.test_x2 = self.tk.text_to_seq(list(self.test_data.question2.values.astype(str)))
        self.test_id = self.test_data.test_id.values.astype(int)

        self.tk.fit_on_seq(self.train_x1)
        self.tk.fit_on_seq(self.train_x2)

        self.tk.fit_on_seq(self.test_x1)
        self.tk.fit_on_seq(self.test_x2)

    def process_with_tokenizer(self):
        print("Text to Sequence...")
        # TODO use column names to get the data, thus making this more generic
        self.train_x1 = self.tk.seq_to_vec(self.train_x1)
        self.train_x2 = self.tk.seq_to_vec(self.train_x2)
        print("Deleting in memory train data...")
        del self.train_data

        # TODO use column names to get the data, thus making this more generic
        self.test_x1 = self.tk.seq_to_vec(self.test_x1)
        self.test_x2 = self.tk.seq_to_vec(self.test_x2)

        self.train_x1 = sequence.pad_sequences(self.train_x1, maxlen=Config.MAX_SEQ_LEN)
        self.train_x2 = sequence.pad_sequences(self.train_x2, maxlen=Config.MAX_SEQ_LEN)

        self.test_x1 = sequence.pad_sequences(self.test_x1, maxlen=Config.MAX_SEQ_LEN)
        self.test_x2 = sequence.pad_sequences(self.test_x2, maxlen=Config.MAX_SEQ_LEN)

        del self.test_data
        print("Deleted intermediate test vals...")

        shuffle_indices = np.random.permutation(np.arange(len(self.train_y)))
        self.train_x1 = self.train_x1[shuffle_indices]
        self.train_x2 = self.train_x2[shuffle_indices]
        self.train_y = self.train_y[shuffle_indices]

        dev_idx = -1 * len(self.train_y) * Config.FLAGS.validation_percentage // 100

        #Validation Sets
        self.train_x1, self.validation_x1 = self.train_x1[:dev_idx], self.train_x1[dev_idx:]
        self.train_x2, self.validation_x2 = self.train_x2[:dev_idx], self.train_x2[dev_idx:]
        self.train_y, self.validation_y = self.train_y[:dev_idx], self.train_y[dev_idx:]


    def pickle_data_set(self):

        #TODO Update class static vriables for
        pickle.dump(zip(self.train_x1, self.train_x2, self.train_y), open(self.PICKLE_TRAIN_DATA_SET_FILE, "wb"))
        pickle.dump(zip(self.validation_x1, self.validation_x2, self.validation_y), open(self.PICKLE_VALIDATION_DATA_SET_FILE, "wb"))

        pickle.dump(self.get_word_index, open("word_index.p", "wb"))

        pickle.dump(zip(self.test_x1, self.test_x2, self.test_id), open(self.PICKLE_TEST_DATA_SET_FILE, 'wb'))

    def batch_iter(self, data, batch_size, shuffle=False):
        """
        Generates a batch iterator for a dataset.
        """
        # Is there a better way to do this?
        data = np.asarray(list(zip(*data)))
        data = np.asarray(list(zip(*data)))

        print('Generating batches for data of shape {}.'.format(data.shape))

        data_size = len(data)
        num_batches_per_epoch = int(len(data) / batch_size) + 1
        print('Number of batches per epoch : ', num_batches_per_epoch)

        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

    def train_batch_iter(self, batch_size, shuffle=True):
        return self.batch_iter(pickle.load(open(self.PICKLE_TRAIN_DATA_SET_FILE, "rb")), batch_size, shuffle)

    def val_batch_iter(self, batch_size):
        return self.batch_iter(pickle.load(open(self.PICKLE_VALIDATION_DATA_SET_FILE, "rb")), batch_size, shuffle=True)

    def test_batch_iter(self, batch_size):
        return self.batch_iter(pickle.load(open(self.PICKLE_TEST_DATA_SET_FILE, "rb")), batch_size)



