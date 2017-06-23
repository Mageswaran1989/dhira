import tensorflow as tf
import tensorflow.contrib.rnn
from dhira.models_base import ModelsBase
from tensorflow.python.framework.function import Defun

class SiameseLSTM(ModelsBase):

    def compute_euclidean_distance(self, x, y):
            """
            Computes the euclidean distance between two tensorflow variables
            """

            d = tf.reduce_sum(tf.square(tf.subtract(x, y)), 1)
            return d

    def compute_contrastive_loss(self, left_feature, right_feature, label, margin = 0.3):

        """
        Compute the contrastive loss as in


        L = 0.5 * Y * D^2 + 0.5 * (Y-1) * {max(0, margin - D)}^2

        **Parameters**
         left_feature: First element of the pair
         right_feature: Second element of the pair
         label: Label of the pair (0 or 1)
         margin: Contrastive margin

        **Returns**
         Return the loss operation

        """

        label = tf.to_float(label)
        one = tf.constant(1.0)

        d = self.compute_euclidean_distance(left_feature, right_feature)
        d_sqrt = tf.sqrt(self.compute_euclidean_distance(left_feature, right_feature))
        first_part = tf.multiply(one - label, d)  # (Y-1)*(d)

        max_part = tf.square(tf.maximum(margin - d_sqrt, 0))
        second_part = tf.multiply(label, max_part)  # (Y) * max(margin - d, 0)

        loss = 0.5 * tf.reduce_mean(first_part + second_part)

        return loss

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

        with tf.variable_scope("biRNN_layer", reuse=reuse):
            with tf.name_scope(side):
                if config.basic_lstm:
                    cell = tf.contrib.rnn.BasicLSTMCell(config.lstm_hidden_units,
                                                        initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                        forget_bias=1.0,
                                                        state_is_tuple=True)
                else:
                    cell = tf.contrib.rnn.LSTMCell(config.lstm_hidden_units,
                                                   initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                   forget_bias=1.0,
                                                   state_is_tuple=True)

                if config.attention:
                    cell = tf.contrib.rnn.AttentionCellWrapper(cell,
                                                               state_is_tuple=True,
                                                               attn_length=10)

                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.siamese_keep_prob)
                cell = tf.contrib.rnn.MultiRNNCell([cell] * config.num_layers, state_is_tuple=True)

                # import tensorflow.contrib.rnn.static_bidirectional_rnn
                outputs, f_state, b_state = tf.contrib.rnn.static_bidirectional_rnn(
                    cell_fw=cell,
                    cell_bw=cell,
                    dtype=tf.float32,
                    # initial_state_fw=,
                    # initial_state_bw=,
                    inputs=x)

        # # c -> Memory/Cell State H -> Output of LSTM node
        # #On every step this will be updated
        self.fw_c_state = f_state[0].c
        self.fw_h_state = f_state[0].h
        self.bw_c_state = b_state[0].c
        self.bw_h_state = b_state[0].h


        # print('Siamese hidden state: ', self.fw_h_state)
        # print('Siamese cell state: ', self.fw_c_state)

        f_state = tf.concat([self.fw_c_state, self.fw_h_state], axis=1)
        b_state = tf.concat([self.bw_c_state, self.bw_h_state], axis=1)
        # return tf.concat([self.fw_h_state, self.bw_h_state], axis=1, name = "sentence_embedding")
        # print('lstm output: ', outputs[-1])
        return outputs[-1]


    def contrastive_loss(self, margin, threshold=1e-5):
        """Contrastive loss:
               E = sum(yd^2 + (1-y)max(margin-d, 0)^2) / 2 / N
               d = L2_dist(data1, data2)
           Usage:
               loss = contrastive_loss(1.0)(data1, data2, similarity/labels)
           Note:
               This is a numeric stable version of contrastive loss
        """

        @Defun(tf.float32, tf.float32, tf.float32, tf.float32)
        def backward(data1, data2, similarity, diff):
            with tf.name_scope(values=[data1, data2, similarity], name="ContrastiveLoss_grad", default_name="ContrastiveLoss_grad"):
                d_ = data1 - data2
                d_square = tf.reduce_sum(tf.square(d_), 1)
                d = tf.sqrt(d_square)

                minus = margin - d
                right_diff = minus / (d + threshold)
                right_diff = d_ * tf.reshape(right_diff * tf.to_float(tf.greater(minus, 0)), [-1, 1])

                batch_size = tf.to_float(tf.slice(tf.shape(data1), [0], [1]))
                data1_diff = diff * ((d_ + right_diff) * tf.reshape(similarity, [-1, 1]) - right_diff) / batch_size
                data2_diff = -data1_diff
                return data1_diff, data2_diff, tf.zeros_like(similarity)

        @Defun(tf.float32, tf.float32, tf.float32, grad_func=backward)
        def forward(data1, data2, similarity):  # assume similarity shape = (N,)
            with tf.name_scope(values=[data1, data2, similarity], name="ContrastiveLoss", default_name="ContrastiveLoss"):
                d_ = data1 - data2
                d_square = tf.reduce_sum(tf.square(d_), 1, keep_dims=True)
                d = tf.sqrt(d_square, name='pred')
                self.pred = d
                print('---> use this for pred: ', d)

                minus = margin - d
                sim = similarity * d_square
                nao = (1.0 - similarity) * tf.square(tf.maximum(minus, 0))
                return tf.reduce_mean(sim + nao) / 2

        return forward


    def siamese_RNN(self, config, x, side, reuse=False):

        x = self.convert_input_for_static_bidir_rnn(x, config.embedding_dim, config.sequence_length)

        with tf.variable_scope("RNN_layer", reuse=reuse):
            with tf.name_scope(side):
                if config.basic_lstm:
                    cell = tf.contrib.rnn.BasicLSTMCell(config.lstm_hidden_units,
                                                        initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                        forget_bias=1.0,
                                                        state_is_tuple=True)
                else:
                    cell = tf.contrib.rnn.LSTMCell(config.lstm_hidden_units,
                                                   initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                   forget_bias=1.0,
                                                   state_is_tuple=True)

                if config.attention:
                    cell = tf.contrib.rnn.AttentionCellWrapper(cell,
                                                               state_is_tuple=True,
                                                               attn_length=10)

                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.siamese_keep_prob)
                cell = tf.contrib.rnn.MultiRNNCell([cell] * config.num_layers, state_is_tuple=True)

                # Getting an initial state of all zeros
                initial_state = cell.zero_state(self.batch_size, tf.float32)

                outputs, final_state = tf.contrib.rnn.static_rnn(cell, x,
                                                         initial_state=initial_state)

        return outputs[-1]


    # Reference https://github.com/nfmcclure/tensorflow_cookbook/blob/master/09_Recurrent_Neural_Networks/06_Training_A_Siamese_Similarity_Measure/siamese_similarity_model.py
    def siamese_biRNN_squased_outputs(self, config, x, side, reuse=False):

        x = self.convert_input_for_static_bidir_rnn(x, config.embedding_dim, config.sequence_length)

        with tf.variable_scope("biRNN_layer", reuse=reuse):
            with tf.name_scope(side):
                lstm_forward_cell = tf.contrib.rnn.BasicLSTMCell(config.lstm_hidden_units,
                                                        initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                        forget_bias=1.0,
                                                        state_is_tuple=True)

                lstm_forward_cell = tf.contrib.rnn.DropoutWrapper(lstm_forward_cell, output_keep_prob=config.siamese_keep_prob)

                lstm_backward_cell = tf.contrib.rnn.BasicLSTMCell(config.lstm_hidden_units,
                                                        initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                        forget_bias=1.0,
                                                        state_is_tuple=True)

                lstm_backward_cell = tf.contrib.rnn.DropoutWrapper(lstm_backward_cell, output_keep_prob=config.siamese_keep_prob)


                # import tensorflow.contrib.rnn.static_bidirectional_rnn
                outputs, f_state, b_state = tf.contrib.rnn.static_bidirectional_rnn(
                    cell_fw=lstm_forward_cell,
                    cell_bw=lstm_backward_cell,
                    dtype=tf.float32,
                    # initial_state_fw=,
                    # initial_state_bw=,
                    inputs=x)

        # Average The output over the sequence
        temporal_mean = tf.add_n(outputs) / config.sequence_length

        # Fully connected layer
        output_size = 2
        A = tf.get_variable(name="A", shape=[2 * config.lstm_hidden_units, output_size],
                            dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.1))

        self.add_hist_summary(A)

        b = tf.get_variable(name="b", shape=[output_size], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.1))

        self.add_scalar_summary(b)

        final_output = tf.matmul(temporal_mean, A) + b
        final_output = tf.nn.dropout(final_output, config.siamese_keep_prob)

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
        self.batch_size = batch_size
        self.enable_summary = enable_summary
        self.input_x1 = None
        self.input_x2 = None
        self.input_y = None
        self.embedding_placeholder = None
        self.embed1 = None
        self.embed2 = None

        self.logits = None
        self.cross_entropy = None
        self.pred = None
        self.loss = None
        self.accuracy = None

        with tf.name_scope("input_layer"):
            # Placeholders for input, output and dropout
            self.input_x1 = tf.placeholder(tf.int32,
                                           [None, config.sequence_length], name="input_x1")

            self.input_x2 = tf.placeholder(tf.int32,
                                           [None, config.sequence_length], name="input_x2")

            self.input_y = tf.placeholder(tf.int32, [None], name="input_y") #Shape is pre defined inorder for it get used in summary

            self.add_hist_summary(self.input_y)

        with tf.name_scope("embed_layer"), tf.device('/cpu:0'):
            word_2_vec = tf.Variable(tf.constant(0.0, shape=[vocab_size, config.embedding_dim]),
                            trainable=False, name="word_embeddings")
            # print(word_2_vec)
            self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, config.embedding_dim],
                                                        name="embedding_placeholder")
            embedding_init = word_2_vec.assign(self.embedding_placeholder)

            self.embed1 = tf.nn.embedding_lookup(embedding_init, self.input_x1, name="embed1")
            self.embed2 = tf.nn.embedding_lookup(embedding_init, self.input_x2, name="embed2")

            #Setup summary
            self.add_embeddings(word_2_vec.name, tsv_file_name="word_embeddings_projector")


        ########################################################################################################3

        # with tf.name_scope("birnn_layer"):
        #     self.out1 = self.siamese_biRNN(config, x=self.embed1, side="left", reuse=False)
        #     self.out2 = self.siamese_biRNN(config, x=self.embed2, side="right", reuse=True)
        #
        #     print(self.out1)


        # Using contrasive loss
        # with tf.name_scope("pred_layer"):
        #     self.loss = self.contrastive_loss(1.0)(self.out1, self.out2, tf.to_float(self.input_y),
        #                                                       name='contrasive-loss')
        #
        #     self.add_hist_summary(self.pred)
        #
        #
        # with tf.name_scope("accuracy"):
        #     correct_predictions = tf.equal(tf.round(self.pred), tf.cast(self.input_y, tf.float32))
        #     self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        #
        #     self.add_scalar_summary(self.accuracy)

        ########################################################################################################3

        # # BiDirectionalRNN with hidden layers
        # with tf.name_scope("birnn_layer"):
        #     self.out1 = self.siamese_biRNN(config, x=self.embed1, side="left", reuse=False)
        #     self.out2 = self.siamese_biRNN(config, x=self.embed2, side="right", reuse=True)
        #
        #     print(self.out1)
        #
        # with tf.name_scope("post_siamese_layer"):
        #     if config.multiply:
        #         features = tf.concat([self.out1, self.out2, self.out1 - self.out2,
        #                               tf.multiply(self.out1, self.out2)], axis=1)
        #     else:
        #         features = tf.concat([self.out1, self.out2, self.out1 - self.out2],
        #                              axis=1)
        #
        # with tf.name_scope("concat_layer"):
        #     drop_out_layer = tf.nn.dropout(features, self.dropout_keep_prob)
        #
        #     self.h1 = tf.layers.dense(inputs=drop_out_layer,
        #                               units=config.dense_units,
        #                               name="hidden_layer_1",
        #                               activation=tf.nn.relu,
        #                               kernel_regularizer=tf.contrib.layers.l2_regularizer(config.l2_reg_lambda))
        #
        #     drop_out_layer = tf.nn.dropout(self.h1, self.dropout_keep_prob)
        #
        #     self.h2 = tf.layers.dense(inputs=drop_out_layer,
        #                               units=config.dense_units/2,
        #                               name="hidden_layer_2",
        #                               activation=tf.nn.relu,
        #                               kernel_regularizer=tf.contrib.layers.l2_regularizer(config.l2_reg_lambda))
        #
        #     drop_out_layer = tf.nn.dropout(self.h2, self.dropout_keep_prob)
        #
        # with tf.name_scope("pred_layer"):
        #     self.logits = tf.layers.dense(inputs=drop_out_layer,
        #                                   units=2,
        #                                   activation=tf.nn.relu,
        #                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(config.l2_reg_lambda),
        #                                   name="logits")
        #
        #     self.pred = tf.nn.softmax(self.logits, name="pred")
        #
        #     if (self.enable_summary): self.add_hist_summary(self.pred)
        #     if (self.enable_summary): self.add_hist_summary(self.logits)
        #
        # with tf.name_scope("loss"):
        #     # self.loss = self.contrastive_loss(self.input_y, self.distance, batch_size)
        #
        #     self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
        #                                                                    labels=self.input_y)
        #     self.loss = tf.reduce_mean(self.cross_entropy, name="xentropy_loss")
        #
        # with tf.name_scope("accuracy"):
        #     # correct_prediction = tf.to_int32(tf.nn.in_top_k(self.logits, self.input_y, 1))
        #
        #     correct_predictions = tf.equal(tf.cast(tf.argmax(self.logits, 1, name="total_pred"), tf.int32), self.input_y)
        #     self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        #
        #     if (self.enable_summary): self.add_scalar_summary(self.accuracy)


       ########################################################################################################3

        # # Siamese RNN
        # with tf.name_scope("rnn_layer"):
        #     self.out1 = self.siamese_RNN(config, x=self.embed1, side="left", reuse=False)
        #     self.out2 = self.siamese_RNN(config, x=self.embed2, side="right", reuse=True)
        #
        #     print(self.out1)
        #
        # with tf.name_scope("pred_layer"):
        #
        #     self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.out1, self.out2)), 1, keep_dims=True))
        #     self.distance = tf.div(self.distance,
        #                            tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.out1), 1, keep_dims=True)),
        #                                   tf.sqrt(tf.reduce_sum(tf.square(self.out2), 1, keep_dims=True))))
        #     self.pred = tf.reshape(self.distance, [-1], name="pred")
        #
        #     if (self.enable_summary): self.add_hist_summary(self.pred)
        #
        # with tf.name_scope("loss"):
        #
        #     self.loss = self.contrastive_loss_original(y=self.input_y, distance=self.pred)
        #
        # with tf.name_scope("accuracy"):
        #     correct_predictions = tf.equal(tf.ceil(self.pred), tf.cast(self.input_y, 'float'))
        #     self.accuracy = tf.reduce_mean(tf.cast(correct_predictions,'float'), name="accuracy")
        #
        #     if (self.enable_summary): self.add_scalar_summary(self.accuracy)

        ######################################################################################################

        # BiDirectionalRNN Reference https://github.com/nfmcclure/tensorflow_cookbook/blob/master/09_Recurrent_Neural_Networks/06_Training_A_Siamese_Similarity_Measure/siamese_similarity_model.py

        with tf.name_scope("birnn_layer"):
            self.out1 = self.siamese_biRNN(config, x=self.embed1, side="left", reuse=False)
            self.out2 = self.siamese_biRNN(config, x=self.embed2, side="right", reuse=True)

            print(self.out1)

            # Unit normalize the outputs
            output1 = tf.nn.l2_normalize(self.out1, 1)
            output2 = tf.nn.l2_normalize(self.out2, 1)
            # Return cosine distance
            #   in this case, the dot product of the norms is the same.
            dot_prod = tf.reduce_sum(tf.multiply(output1, output2), 1)

            return dot_prod

    @property
    def get_predictions(self, model_output):
        predictions = tf.sign(model_output, name="predictions")
        return (predictions)

    @property
    def loss(self, model_output, target_labels):
        print("Returing SiameseLSTM loss")
        return None

    @property
    def accuracy(self, model_output, target_labels):
        return None

    @property
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

    @property
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
