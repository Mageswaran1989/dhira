import logging

import tensorflow as tf
from overrides import overrides
from tensorflow.contrib.rnn import LSTMCell

from dhira.tf.models.internal.base_tf_model import BaseTFModel
from dhira.tf.models.util.rnn import SwitchableDropoutWrapper, last_relevant_output, mean_pool

logger = logging.getLogger(__name__)

class LSTMSentimentAnalysis(BaseTFModel):
    def __init__(self,
                 run_id,
                 word_vocab_size,
                 word_embedding_dim,
                 word_embedding_matrix,
                 fine_tune_embeddings,
                 sequence_legth,
                 rnn_output_mode,
                 output_keep_prob,
                 rnn_hidden_size,
                 num_lstm_layers,
                 batch_size,
                 learning_rate = 0.001,
                 name='LSTMSentimentAnalysis',
                 save_dir=None,
                 log_dir=None):
        """
        Create a model LSTM model to predict the mood of the sentence as positive or negative.

        :param word_vocab_size: int
            The number of unique tokens in the dataset, plus the UNK and padding
            tokens. Alternatively, the highest index assigned to any word, +1.
            This is used by the model to figure out the dimensionality of the
            embedding matrix.

        :param word_embedding_dim: int
            The length of a word embedding. This is used by
            the model to figure out the dimensionality of the embedding matrix.

        :param word_embedding_matrix: numpy array, optional if predicting
            A numpy array of shape (word_vocab_size, word_emb_dim).
            word_embedding_matrix[index] should represent the word vector for
            that particular word index. This is used to initialize the
            word embedding matrix in the model, and is optional if predicting
            since we assume that the word embeddings variable will be loaded
            with the model.

        :param fine_tune_embeddings: boolean
            If true, sets the embeddings to be trainable.

        :param rnn_hidden_size: int
            The output dimension of the RNN encoder. Note that this model uses a
            bidirectional LSTM, so the actual sentence vectors will be
            of length 2*rnn_hidden_size.

        :param share_encoder_weights: boolean
            Whether to use the same encoder on both input sentnces (thus
            sharing weights), or a different one for each sentence.

        :param rnn_output_mode: str
            How to calculate the final sentence representation from the RNN
            outputs. mean pool" indicates that the outputs will be averaged (with
            respect to padding), and "last" indicates that the last
            relevant output will be used as the sentence representation.

        :param output_keep_prob: float
            The probability of keeping an RNN outputs to keep, as opposed
            to dropping it out.
        """
        super(LSTMSentimentAnalysis, self).__init__(name=name,
                                            run_id=run_id,
                                            save_dir=save_dir,
                                            log_dir=log_dir)
        self.word_vocab_size = word_vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.word_embedding_matrix = word_embedding_matrix
        self.fine_tune_embeddings = fine_tune_embeddings
        self.rnn_hidden_size = rnn_hidden_size
        self.sequence_legth = sequence_legth
        self.rnn_output_mode = rnn_output_mode
        self.output_keep_prob = output_keep_prob
        self.num_lstm_layers = num_lstm_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    @overrides
    def _create_placeholders(self):
        #[batch_size, sequence_length]
        self.sentence = tf.placeholder(tf.int32, [None, self.sequence_legth], name='inputs')
        #[batch_size, num_labels]
        self.labels = tf.placeholder(tf.int32, [None, 2], name='labels')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        # A boolean that encodes whether we are training or evaluating
        self.is_train = tf.placeholder('bool', [], name='is_train')

    @overrides
    def _setup_graph_def(self):

        #Create a Tensorflow Embedding Matrix for lookup
        with tf.variable_scope("embeddings"), tf.device('/cpu:0'):
            with tf.variable_scope("embedding_var"), tf.device("/cpu:0"):
                if self._mode == "train":
                    # Load the word embedding matrix that was passed in
                    # since we are training

                    # [vocab_size, embedd_dim]
                    word_emb_mat = tf.get_variable(
                        "word_emb_mat",
                        dtype="float",
                        shape=[self.word_vocab_size,
                               self.word_embedding_dim],
                        initializer=tf.constant_initializer(
                            self.word_embedding_matrix),
                        trainable=self.fine_tune_embeddings)
                else:
                    # We are not training, so a model should have been
                    # loaded with the embedding matrix already there.
                    word_emb_mat = tf.get_variable("word_emb_mat",
                                                   shape=[self.word_vocab_size,
                                                          self.word_embedding_dim],
                                                   dtype="float",
                                                   trainable=self.fine_tune_embeddings)
        logger.info('word_emb_mat: ------> {}'.format(word_emb_mat))
        with tf.variable_scope("embeddings"), tf.device('/gpu:0'):
            with tf.variable_scope("word_embeddings"):
                # Shape: [batch_size, num_sentence_words, embedding_dim]
                word_embedded_sentence = tf.nn.embedding_lookup(word_emb_mat,
                    self.sentence)

            logger.info('word_embedded_sentence: ------> {}'.format(word_embedded_sentence))
        # LSTM cell
        lstm = tf.contrib.rnn.LSTMCell(self.rnn_hidden_size, state_is_tuple=True)
        logger.info('lstm: ------> {}'.format(lstm))

        # Add dropout to the cell
        drop =  SwitchableDropoutWrapper(
                    lstm,
                    self.is_train,
                    output_keep_prob=self.output_keep_prob)

        # Stack up multiple LSTM layers, for deep learning
        cell = tf.contrib.rnn.MultiRNNCell([drop] * self.num_lstm_layers)
        logger.info('cell: ------> {}'.format(cell))

        # Getting an initial state of all zeros
        #[batch_size, rnn_hidden_size]
        initial_state = cell.zero_state(self.batch_size, tf.float32)
        logger.info('initial_state: ------> {}'.format(initial_state))

        outputs, final_state = tf.nn.dynamic_rnn(cell, word_embedded_sentence, dtype=tf.float32)
                                                 # initial_state=initial_state)
        logger.info('outputs: -----> {}'.format(outputs))
        logger.info('final_state: -----> {}'.format(final_state))

        # [batch_size, 1]
        self.predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 2,
                                                             activation_fn=tf.sigmoid)
        logger.info('self.labels: ------> {}'.format(self.labels))
        self.loss = tf.losses.mean_squared_error(self.labels, self.predictions)
        logger.info('self.loss: ------> {}'.format(self.loss))

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        # correct_pred = tf.equal(tf.cast(tf.round(self.predictions), tf.int32), self.labels)
        correct_pred = tf.equal(tf.argmax(self.predictions, 1), tf.argmax(self.labels, 1))

        self.eval_operation = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        logger.info('self.eval_operation: ------> {}'.format(self.eval_operation))

    @overrides
    def _get_train_feed_dict(self, batch):
        tokens, labels = batch
        feed_dict = {self.sentence: tokens[0],
                     self.labels: labels[0],
                     self.is_train: True}
        return feed_dict

    @overrides
    def _get_validation_feed_dict(self, batch):
        tokens, labels = batch
        feed_dict = {self.sentence: tokens[0],
                     self.labels: labels[0],
                     self.is_train: False}
        return feed_dict

    @overrides
    def _get_test_feed_dict(self, batch):
        tokens, labels = batch
        feed_dict = {self.sentence: tokens[0],
                     self.is_train: False}
        return feed_dict