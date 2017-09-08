import logging

import tensorflow as tf
from overrides import overrides

from dhira.tf.models.internal.base_tf_model import BaseTFModel

logger = logging.getLogger(__name__)

class SentimentCNN(BaseTFModel):

    @overrides
    def __init__(self,
                 save_dir,
                 log_dir,
                 run_id,
                 word_vocab_size,
                 word_embedding_dim,
                 word_embedding_matrix,
                 keep_prob = 0.5,
                 filter_sizes=[3,4,5],
                 num_filters=128,
                 l2_reg_lambda=0.1,
                 sequence_length = 59,
                 num_classes = 2,
                 name='sentiment_cnn'):
        """
        
        :param save_dir: 
        :param log_dir: 
        :param run_id: 
        :param word_vocab_size: The size of our vocabulary. This is needed to define the size of our embedding layer, 
                which will have shape [vocabulary_size, embedding_size].
        :param word_embedding_dim: The dimensionality of our embeddings.
        :param keep_prob: 
        :param filter_sizes: The number of words we want our convolutional filters to cover. We will have num_filters 
            for each size specified here. For example, [3, 4, 5] means that we will have filters that slide over 
            3, 4 and 5 words respectively, for a total of 3 * num_filters filters.
        :param num_filters: The number of filters per filter size
        :param l2_reg_lambda: 
        :param sequence_length: The length of our sentences. 
                        Remember that we padded all our sentences to have the same length (59 for our data set).
        :param num_classes: Number of classes in the output layer, two in our case (positive and negative).
        :param name: 
        """
        super(SentimentCNN, self).__init__(name=name,
                                              run_id=run_id,
                                              save_dir=save_dir,
                                              log_dir=log_dir)
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.vocab_size = word_vocab_size
        self.embedding_size = word_embedding_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.l2_reg_lambda = l2_reg_lambda
        self.keep_prob = keep_prob
        self.embedding_matrix = word_embedding_matrix

        self.W = None


    @overrides
    def _create_placeholders(self):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    @overrides
    def _setup_graph_def(self):
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # self.word_emb_mat = tf.Variable(
            #     tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
            #     name="W")

            self.word_emb_mat = tf.get_variable("word_emb_mat",
                                                dtype="float",
                                                shape=[self.vocab_size, self.embedding_size],
                                                initializer=tf.constant_initializer(self.embedding_matrix),
                                                trainable=False)

            self.word_embedded_tokens = tf.nn.embedding_lookup(self.word_emb_mat, self.input_x)
            #[None, Sequence Length, Embedding Size, 1] i.e [?,59,300,1]
            self.word_embedded_tokens_expanded = tf.expand_dims(self.word_embedded_tokens, -1)

        # Creates  convolution layer -> relu -> maxpool layer for each filter size eg:[3,4,5]
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.word_embedded_tokens_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity activation function
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.sigmoid(self.scores)#tf.argmax(self.scores, 1, name="predictions")
            self._y_pred = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self._y_pred, tf.argmax(self.input_y, 1))
            self.eval_operation = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.training_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        # # Keep track of gradient values and sparsity (optional)
        # grad_summaries = []
        # for g, v in grads_and_vars:
        #     if g is not None:
        #         grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
        #         sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
        #         grad_summaries.append(grad_hist_summary)
        #         grad_summaries.append(sparsity_summary)

        # self._add_scalar_summary(self.loss)
        # self._add_scalar_summary(self.eval_operation)

        # self._add_hist_summary(grad_summaries)

    @overrides
    def _get_train_feed_dict(self, batch):
        tokens, labels = batch
        feed_dict = {self.input_x: tokens[0],
                     self.input_y: labels[0],
                     self.dropout_keep_prob: self.keep_prob}
        return feed_dict

    @overrides
    def _get_validation_feed_dict(self, batch):
        tokens, labels = batch
        feed_dict = {self.input_x: tokens[0],
                     self.input_y: labels[0],
                     self.dropout_keep_prob: 1.0}
        return feed_dict

    @overrides
    def _get_test_feed_dict(self, batch):
        tokens, labels = batch
        feed_dict = {self.input_x: tokens[0],
                     self.dropout_keep_prob: 1.0}
        return feed_dict