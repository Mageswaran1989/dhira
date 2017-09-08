from __future__ import division #for 3/4
import logging

import tensorflow as tf
from overrides import overrides

from dhira.tf.models.internal.base_tf_model import BaseTFModel

logger = logging.getLogger(__name__)

class NotTrainedError(Exception):
    pass


class Glove(BaseTFModel):
    """
    Glove Model to train your own embedding matrix using Glove Paper
    
    Use GloveDataset and GloveFeature
    
    Config Parameters:
        vocabulary_size
        min_occurrences
        left_size
        right_size
    """
    @overrides
    def __init__(self, name, save_dir, log_dir, run_id,
                 embedding_size, cooccurrence_cap, vocabulary_size, batch_size, learning_rate):
        super(Glove, self).__init__(name=name,
                                              run_id=run_id,
                                              save_dir=save_dir,
                                              log_dir=log_dir)

        self.embedding_size: int = embedding_size
        # self.left_context: int = config_dict.pop("left_context")
        # self.right_context: int = config_dict.pop("right_context")

        # self.max_vocab_size = config_dict.pop("max_vocab_size")
        # self.min_occurrences = config_dict.pop("min_occurrences")
        self.cooccurrence_cap = cooccurrence_cap
        self.vocab_size = vocabulary_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.scaling_factor = float(3/4) #As per the Glove Paper
        self.embeddings = None

    @overrides
    def _create_placeholders(self):
        self.focal_input = tf.placeholder(tf.int32, shape=None, name="focal_words")
        self.context_input = tf.placeholder(tf.int32, shape=None, name="context_words")
        self.cooccurrence_count = tf.placeholder(tf.float32, shape=None, name="cooccurrence_count")

    @overrides
    def _setup_graph_def(self):
        count_max = tf.constant([self.cooccurrence_cap], dtype=tf.float32, name='max_cooccurrence_count')
        scaling_factor = tf.constant([self.scaling_factor], dtype=tf.float32, name="scaling_factor")

        #[vocab_size x embedding_size] w
        focal_embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], 1.0, -1.0),
                                        name="focal_embeddings")
        # [vocab_size x embedding_size] w˜
        context_embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], 1.0, -1.0),
                                            name="context_embeddings")

        # [vocab_size]
        # b_i {i : 0 to Vocab Size}
        focal_biases = tf.Variable(tf.random_uniform([self.vocab_size], 1.0, -1.0), name='focal_biases')
        # b_j˜ {j : 0 to Vocab Size}
        context_biases = tf.Variable(tf.random_uniform([self.vocab_size], 1.0, -1.0), name="context_biases")

        focal_embedding = tf.nn.embedding_lookup([focal_embeddings], self.focal_input)
        context_embedding = tf.nn.embedding_lookup([context_embeddings], self.context_input)
        focal_bias = tf.nn.embedding_lookup([focal_biases], self.focal_input)
        context_bias = tf.nn.embedding_lookup([context_biases], self.context_input)

        # f(x) = min(1, (count/count_max)^scaling_factor)
        weighting_factor = tf.minimum(1.0, tf.pow(tf.div(self.cooccurrence_count, count_max),
                                                scaling_factor))

        # W^t W˜
        embedding_product = tf.reduce_sum(tf.multiply(focal_embedding, context_embedding), 1)

        # log(X_ij)
        log_cooccurrences = tf.log(tf.to_float(self.cooccurrence_count))

        # W^t W˜ + b_i + b_j˜ - log(X_ij)
        distance_expr = tf.square(tf.add_n([
            embedding_product,
            focal_bias,
            context_bias,
            tf.negative(log_cooccurrences)]))

        single_losses = tf.multiply(weighting_factor, distance_expr)
        #J
        self.loss = tf.reduce_sum(single_losses, name="GloVe_loss")

        self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)
        self.__combined_embeddings = tf.add(focal_embeddings, context_embeddings, name="combined_embeddings")
        self.eval_operation = tf.constant(0, name='dummy_accuracy')

        self.predictions = self.__combined_embeddings #Dummy assignment ignore TODO handle this is in Base class

    @overrides
    def _evaluate_model_parameters(self, session):
        self.embeddings = self.__combined_embeddings.eval(session=session)
        self.predictions = self.embeddings
    @overrides
    def _get_train_feed_dict(self, batch):
        inputs, target = batch
        focal, context, count = inputs
        feed_dict = {
            self.focal_input: focal,
            self.context_input : context,
            self.cooccurrence_count : count
        }
        return feed_dict

    @overrides
    def _get_validation_feed_dict(self, batch):
        inputs, target = batch
        focal, context, count = inputs
        feed_dict = {
            self.focal_input: focal,
            self.context_input : context,
            self.cooccurrence_count : count
        }
        return feed_dict

    @overrides
    def _get_test_feed_dict(self, batch):
        ''