import tensorflow as tf

tf.app.flags.DEFINE_float('default_keep_prob', 0.5, " ")
tf.app.flags.DEFINE_string("optimizer", "adam", "")
tf.app.flags.DEFINE_integer("validation_percentage", 10, "")


#Siamese Network
tf.app.flags.DEFINE_integer('num_layers', 1, "Number of LSTM Layers")
tf.app.flags.DEFINE_integer('sequence_length', 30, " ")
tf.app.flags.DEFINE_integer('embedding_dim', 300, " ")
tf.app.flags.DEFINE_integer('lstm_hidden_units', 256, " ")
tf.app.flags.DEFINE_integer('dense_units', 128, " ")
tf.app.flags.DEFINE_integer('l2_reg_lambda', 1e-3, " ")
tf.app.flags.DEFINE_integer('evaluate_every', 2, " ")
tf.app.flags.DEFINE_float('siamese_keep_prob', 0.25, " ")
tf.app.flags.DEFINE_float('lr', 0.002, " ")
tf.app.flags.DEFINE_boolean('early_stopping', True, " ")
tf.app.flags.DEFINE_boolean('multiply',  True, " ")
tf.app.flags.DEFINE_boolean('attention',  False, " ")
tf.app.flags.DEFINE_boolean('basic_lstm',  False, " ")
tf.app.flags.DEFINE_integer('ignore_one_in_every', 3, " ")

#  Text Convolutional Network
tf.app.flags.DEFINE_integer('emb_size', 300, 'Size of word embeddings')
tf.app.flags.DEFINE_integer('num_kernel', 100, 'Number of filters for each window size')
tf.app.flags.DEFINE_integer('min_window', 3, 'Minimum size of filter window')
tf.app.flags.DEFINE_integer('max_window', 5, 'Maximum size of filter window')
tf.app.flags.DEFINE_integer('vocab_size', 20000, 'Vocabulary size')
tf.app.flags.DEFINE_integer('num_class', 2, 'Number of class to consider')
tf.app.flags.DEFINE_integer('sent_len', 56, 'Input sentence length. This is after the padding is performed.')
tf.app.flags.DEFINE_float('l2_reg', 1e-3, 'l2 regularization weight')
tf.app.flags.DEFINE_float('conv_keep_prob', 0.5, 'Dropout rate. 0 is no dropout.')


##########################################################################

from enum import Enum
class NLPEngine(Enum):
    NLTK = 'nltk'
    SPACY = 'spacy'
    NONE = 'none'

###########################################################################

class Config:
    FLAGS = tf.app.flags.FLAGS

    BASE_DIR = 'data/'
    EMBEDDING_PATH = BASE_DIR + 'word2vec.840B.300d.txt'
    TRAIN_DATA_PATH = BASE_DIR + 'train.csv'
    TEST_DATA_PATH = BASE_DIR + 'test.csv'
    VOCAB_SIZE = 200000
    MAX_SEQ_LEN = 30

    NLP_BACK_END = SupportedNLPBackEnd.SPACY

    PREPROCESS_ENABLED = False

