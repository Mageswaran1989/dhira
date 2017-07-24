from copy import deepcopy
import logging
from overrides import overrides


import tensorflow as tf
from dhira.tf.models.base_tf_model import BaseTFModel

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
    def __init__(self, config_dict):
        config_dict = deepcopy(config_dict)
        mode = config_dict.pop("mode")
        super(Glove, self).__init__(mode=mode)

        self.embedding_size: int = config_dict.pop("embedding_size")
        # self.left_context: int = config_dict.pop("left_context")
        # self.right_context: int = config_dict.pop("right_context")

        # self.max_vocab_size = config_dict.pop("max_vocab_size")
        # self.min_occurrences = config_dict.pop("min_occurrences")
        self.cooccurrence_cap = config_dict.pop("cooccurrence_cap", None)
        self.vocab_size = config_dict.pop("vocabulary_size", None)
        self.batch_size = config_dict.pop("batch_size")
        self.learning_rate = config_dict.pop("learning_rate")
        self.scaling_factor = float(3/4)
        self.__embeddings = None

        if config_dict:
            logger.warning("UNUSED VALUES IN CONFIG DICT: {}".format(config_dict))

    @overrides
    def _create_placeholders(self):
        self.focal_input = tf.placeholder(tf.int32, shape=[self.batch_size], name="focal_words")
        self.context_input = tf.placeholder(tf.int32, shape=[self.batch_size], name="context_words")
        self.cooccurrence_count = tf.placeholder(tf.float32, shape=[self.batch_size], name="cooccurrence_count")

    @overrides
    def _build_forward(self):
        count_max = tf.constant([self.cooccurrence_cap], dtype=tf.float32, name='max_cooccurrence_count')
        scaling_factor = tf.constant([self.scaling_factor], dtype=tf.float32, name="scaling_factor")

        #[vocab_size x embedding_size]
        focal_embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], 1.0, -1.0),
                                        name="focal_embeddings")
        # [vocab_size x embedding_size]
        context_embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], 1.0, -1.0),
                                            name="context_embeddings")

        focal_biases = tf.Variable(tf.random_uniform([self.vocab_size], 1.0, -1.0), name='focal_biases')
        context_biases = tf.Variable(tf.random_uniform([self.vocab_size], 1.0, -1.0), name="context_biases")

        focal_embedding = tf.nn.embedding_lookup([focal_embeddings], self.focal_input)
        context_embedding = tf.nn.embedding_lookup([context_embeddings], self.context_input)

        focal_bias = tf.nn.embedding_lookup([focal_biases], self.focal_input)
        context_bias = tf.nn.embedding_lookup([context_biases], self.context_input)

        weighting_factor = tf.minimum(1.0, tf.pow(tf.div(self.cooccurrence_count, count_max),
                                                scaling_factor))

        embedding_product = tf.reduce_sum(tf.multiply(focal_embedding, context_embedding), 1)

        log_cooccurrences = tf.log(tf.to_float(self.cooccurrence_count))

        distance_expr = tf.square(tf.add_n([
            embedding_product,
            focal_bias,
            context_bias,
            tf.negative(log_cooccurrences)]))

        single_losses = tf.multiply(weighting_factor, distance_expr)
        self.loss = tf.reduce_sum(single_losses, name="GloVe_loss")
        self.training_op = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)
        self.__combined_embeddings = tf.add(focal_embeddings, context_embeddings, name="combined_embeddings")
        self.accuracy = tf.constant(0, name='dummy_accuracy')
        self.add_scalar_summary(self.loss)

    @overrides
    def _evaluate_model_parameters(self, session):
        self.__embeddings = self.__combined_embeddings.eval(session=session)

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

    def embedding_for(self, word_str_or_id):
        if isinstance(word_str_or_id, str):
            return self.__embeddings[self.__word_to_id[word_str_or_id]]
        elif isinstance(word_str_or_id, int):
            return self.__embeddings[word_str_or_id]