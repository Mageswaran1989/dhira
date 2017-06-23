import tensorflow as tf
import tensorflow.contrib.rnn
from dhira.models_base import ModelsBase
from tensorflow.python.framework.function import Defun



class SiameseLSTM(ModelsBase):

    @staticmethod
    def convert_input_for_static_bidir_rnn(x, embedding_size, sequence_length):
        # Prepare data shape to match `bidirectional_rnn` function requirements
        # Current data input shape: (batch_size, config.sequence_length, embedding_size)
        # Required shape: 'config.sequence_length' tensors list of shape (batch_size, embedding_size)
        # Permuting batch_size and n_steps
        x = tf.transpose(x, [1, 0, 2])
        # Reshape to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, embedding_size])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(x, sequence_length, 0)
        return x

    def siamese_biRNN(self, config, x, side, reuse=False):

        x = self.convert_input_for_static_bidir_rnn(x, config.embedding_dim, config.sequence_length)

        with tf.variable_scope("biDirRNN", reuse=reuse):
            with tf.name_scope(side):
                cell = tf.contrib.rnn.BasicLSTMCell(config.lstm_hidden_units,
                                                        forget_bias=1.0,
                                                        state_is_tuple=True)

                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)

                # import tensorflow.contrib.rnn.static_bidirectional_rnn
                outputs, f_state, b_state = tf.contrib.rnn.static_bidirectional_rnn(
                    cell_fw=cell,
                    cell_bw=cell,
                    dtype=tf.float32,
                    # initial_state_fw=,
                    # initial_state_bw=,
                    inputs=x)

                # outputs , _ = tf.contrib.rnn.static_rnn(cell=cell,inputs=x,dtype=tf.float32,)

                self.add_hist_summary(outputs[-1])

                # return (outputs[-1])

                # Average The output over the sequence
                output_size = config.embedding_dim # WARNING thsi changes the predictions
                temporal_mean = tf.add_n(outputs) / config.sequence_length
                #
                # Fully connected layer

                # A = tf.get_variable(name="A", shape=[2 * config.lstm_hidden_units, output_size],
                #                     dtype=tf.float32,
                #                     initializer=tf.random_normal_initializer(stddev=0.5))
                #
                # self.add_hist_summary(A)
                #
                # b = tf.get_variable(name="b", shape=[output_size], dtype=tf.float32,
                #                     initializer=tf.random_normal_initializer(stddev=0.5))
                #
                # self.add_hist_summary(b)
                #
                # final_output = tf.matmul(temporal_mean, A) + b

                final_output = tf.layers.dense(temporal_mean, output_size,
                                               bias_initializer=tf.random_normal_initializer(stddev=0.05),
                                               kernel_initializer=tf.random_normal_initializer(stddev=0.05),
                                               activation=tf.nn.relu,
                                               name='lstm_dense_layer')

                self.add_hist_summary(final_output)

                final_output = tf.nn.dropout(final_output, self.dropout_keep_prob)

        return (final_output)

    def __init__(self, config, vocab_size, batch_size, enable_summary = True):
        """
        Model is intialized here
        :param config: 
        :param vocab_size: 
        :param batch_size: 
        :param enable_summary: 
        """
        ModelsBase.__init__(self)
        self.config = config
        self.batch_size = batch_size
        self.enable_summary = enable_summary
        self.input_x1 = None
        self.input_x2 = None
        self.input_y = None
        self.embedding_placeholder = None
        self.embed1 = None
        self.embed2 = None
        self.is_training = None

        with tf.name_scope("input-layer"):
            # Placeholders for input, output and dropout
            self.input_x1 = tf.placeholder(tf.int32,
                                           [None, config.sequence_length], name="input_x1")
            print(self.input_x1)
            self.input_x2 = tf.placeholder(tf.int32,
                                           [None, config.sequence_length], name="input_x2")

            self.input_y = tf.placeholder(tf.int32, [None], name="input_y") #Shape is pre defined inorder for it get used in summary

            self.is_training = tf.placeholder(tf.bool, name="is_training")

            self.add_hist_summary(self.input_y)

        with tf.name_scope("embed-layer"), tf.device('/cpu:0'):
            word_2_vec = tf.Variable(tf.constant(0.0, shape=[vocab_size, config.embedding_dim]),
                            trainable=False, name="word_embeddings")
            # print(word_2_vec)
            self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, config.embedding_dim],
                                                        name="embedding_placeholder")

            print(self.embedding_placeholder)
            embedding_init = word_2_vec.assign(self.embedding_placeholder)

            self.embed1 = tf.nn.embedding_lookup(embedding_init, self.input_x1, name="embed1")
            self.embed2 = tf.nn.embedding_lookup(embedding_init, self.input_x2, name="embed2")

            #Setup summary
            self.add_embeddings(word_2_vec.name, tsv_file_name="word_embeddings_projector")

        with tf.name_scope("hidden-layer"):
            self.out1 = self.siamese_biRNN(self.config, x=self.embed1, side="left", reuse=False)
            self.out2 = self.siamese_biRNN(self.config, x=self.embed2, side="right", reuse=True)

            print(self.out1)

            layer = tf.concat([self.out1, self.out2], axis=1, name='concat')

            self.add_hist_summary(layer)

            layer = tf.nn.dropout(layer, self.dropout_keep_prob)
            layer = tf.layers.batch_normalization(layer, training=self.is_training)
            layer = tf.layers.dense(layer, config.dense_units,
                                           bias_initializer=tf.random_normal_initializer(stddev=0.01),
                                           kernel_initializer=tf.random_normal_initializer(stddev=0.05),
                                           # activation=tf.nn.relu,
                                           name='dense_layer_1')
            # Leaky ReLU
            layer = tf.maximum(0.2 * layer, layer)
            self.add_hist_summary(layer)

            layer = tf.nn.dropout(layer, self.dropout_keep_prob)
            layer = tf.layers.batch_normalization(layer, training=self.is_training)
            layer = tf.layers.dense(layer, config.dense_units/2,
                                           bias_initializer=tf.random_normal_initializer(stddev=0.01),
                                           kernel_initializer=tf.random_normal_initializer(stddev=0.05),
                                           # activation=tf.nn.relu,
                                           name='dense_layer_2')
            # Leaky ReLU
            layer = tf.maximum(0.2 * layer, layer)
            self.add_hist_summary(layer)

            with tf.name_scope("ouput-layer"):
                layer = tf.nn.dropout(layer, self.dropout_keep_prob)
                # Unit normalize the outputs
                layer = tf.layers.batch_normalization(layer, training=self.is_training)
                self.model_output = tf.layers.dense(layer, 1,
                                        bias_initializer=tf.random_normal_initializer(stddev=0.01),
                                        kernel_initializer=tf.random_normal_initializer(stddev=0.05),
                                        # activation=tf.nn.sigmoid,
                                        name='dense_layer_3')

                self.add_hist_summary(self.model_output)

                # Return cosine distance
                #   in this case, the dot product of the norms is the same.
                # self.model_output = tf.reduce_sum(tf.multiply(output1, output2), 1) #TODO chain this i.e pipelining

                print('modle output: ', self.model_output)

    def get_predictions(self):
        with tf.name_scope("prediction-layer"):
            if(self.model_output is None):
                raise UserWarning("Build the model before using it!")
            predictions = tf.nn.sigmoid(self.model_output, name="predictions")
            # predictions = tf.identity(self.model_output, name="predictions")
            print(predictions)
            # predictions = tf.sign(self.model_output, name="predictions")
            self.add_hist_summary(predictions)
            return (predictions)

    def loss(self):
        with tf.name_scope("loss-layer"):
            print("Returing SiameseLSTM loss")

            ####################################################

            # loss = tf.nn.log_poisson_loss(log_input=self.model_output,
            #                               targets=tf.cast(tf.expand_dims(self.input_y,1), 'float32'),
            #                               compute_full_loss=False, name='loss')
            # loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.model_output,
            #                                                labels=tf.cast(tf.expand_dims(self.input_y,1), 'float32'),
            #                                                name='loss')
            loss = tf.losses.log_loss(labels=tf.cast(tf.expand_dims(self.input_y,1), 'float32'), predictions=self.get_predictions())
            avg_loss = tf.reduce_mean(loss, name='loss')

            print(avg_loss)

            self.add_scalar_summary(avg_loss)
            return (avg_loss)

            ####################################################
            # tf.nn.sigmoid_cross_entropy_with_logits()

            # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.model_output,
            #                                                                labels=self.input_y)
            # avg_loss = tf.reduce_mean(cross_entropy, name="xentropy_loss")
            #
            # self.add_scalar_summary(avg_loss)
            # return (avg_loss)

            ####################################################

            # Calculate the positive losses
            pos_loss_term = 0.25 * tf.square(tf.subtract(1., self.model_output))

            # If y-target is -1 to 1, then do the following
            # pos_mult = tf.add(tf.multiply(0.5, y_target), 0.5)
            # Else if y-target is 0 to 1, then do the following
            pos_mult = tf.cast(self.input_y, tf.float32)

            # Make sure positive losses are on similar strings
            positive_loss = tf.multiply(pos_mult, pos_loss_term)

            # Calculate negative losses, then make sure on dissimilar strings

            # If y-target is -1 to 1, then do the following:
            # neg_mult = tf.add(tf.mul(-0.5, y_target), 0.5)
            # Else if y-target is 0 to 1, then do the following
            neg_mult = tf.subtract(1., tf.cast(self.input_y, tf.float32))

            negative_loss = neg_mult * tf.square(self.model_output)

            # Combine similar and dissimilar losses
            loss = tf.add(positive_loss, negative_loss)

            # Create the margin term.  This is when the targets are 0.,
            #  and the scores are less than m, return 0.

            # Check if target is zero (dissimilar strings)
            target_zero = tf.equal(tf.cast(self.input_y, tf.float32), 0.)
            # Check if cosine outputs is smaller than margin
            less_than_margin = tf.less(self.model_output, 0.2) #TODO margin config
            # Check if both are true
            both_logical = tf.logical_and(target_zero, less_than_margin)
            both_logical = tf.cast(both_logical, tf.float32)
            # If both are true, then multiply by (1-1)=0.
            multiplicative_factor = tf.cast(1. - both_logical, tf.float32)
            total_loss = tf.multiply(loss, multiplicative_factor)

            # Average loss over batch
            avg_loss = tf.reduce_mean(total_loss, name='loss')

            print(avg_loss)

            self.add_scalar_summary(avg_loss)
            return (avg_loss)

    def accuracy(self):
        predictions = self.get_predictions()

        with tf.name_scope("accuracy-layer"):

            if(len(self.model_output.get_shape()) == 1):

                # Cast into integers (outputs can only be -1 or +1)
                # Change targets from (0,1) --> (-1, 1)
                #    via (2 * x - 1)
                # y_target_int = tf.sub(tf.mul(y_target_int, 2), 1)
                # predictions_int = tf.cast(tf.sign(predictions), tf.int32)
                #
                correct_predictions = tf.equal(tf.cast(tf.greater(predictions,0.5),'int32'), self.input_y)
                accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')
            else:
                correct_predictions = tf.equal(tf.cast(tf.argmax(predictions, 1, name="total_pred"), tf.int32), self.input_y)
                accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

            self.add_scalar_summary(accuracy)
            return (accuracy)

    def image_matrix_3_channels(self):
        with tf.name_scope("to_conv_input_layer"):
            img_size = self.out1.get_shape()[1].value

            channel_1 = ([self.out1 + self.out2])  # tf.concat([self.out1, self.out2], axis=1)
            channel_2 = ([self.out1 - self.out2])
            channel_3 = ([tf.multiply(self.out1, self.out2)])

            # TODO find a way to pregramatically reshape
            channel_1 = tf.reshape(channel_1, [-1, 32, 16]) #in case of LSTM hidden state/ cell count = 256, here 256 + 256
            channel_2 = tf.reshape(channel_2, [-1, 32, 16])
            channel_3 = tf.reshape(channel_3, [-1, 32, 16])

            merged_channels = tf.stack([channel_1, channel_2, channel_3], axis=3, name="merged_siamease_hidden_states")

            # print(self.out1)
            # print(self.out2)
            # print(merged_channels)

            image = tf.summary.image("Siamese-image_matrix_3_channels", merged_channels)
            self.train_summaries.append(image)
            self.val_summaries.append(image)

            return merged_channels

    def image_matrix_4_channels(self):
        with tf.name_scope("siamese_output_layer"):
            channel_1 = tf.reshape(self.fw_c_state, [-1, 32, 16])  #in case of LSTM hidden state/ cell count = 512
            channel_2 = tf.reshape(self.fw_h_state, [-1, 32, 16])
            channel_3 = tf.reshape(self.bw_c_state, [-1, 32, 16])
            channel_4 = tf.reshape(self.bw_h_state, [-1, 32, 16])
            merged_channels = tf.stack([channel_1, channel_2, channel_3, channel_4], axis=3, name="merged_siamease_hidden_states")
            # print('4d channel ', merged_channels)

            #This by default will be added to sumaaries when called
            image = tf.summary.image("Siamese-image_matrix_4_channels", merged_channels, max_outputs=4)
            self.train_summaries.append(image)
            self.val_summaries.append(image)

            return merged_channels
