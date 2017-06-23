import tensorflow as tf
import tensorflow.contrib.layers
from dhira.models_base import ModelsBase


def _variable_on_cpu(name, shape, initializer):
    with tf.device("/cpu:0"):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

def _variable_with_weight_decay(name, shape, initializer, wd):
    var = _variable_on_cpu(name, shape, initializer)
    if wd is not None and wd != 0.:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name="weight_loss")
    else:
        weight_decay = tf.constant(0.0, dtype=tf.float32)
    return var, weight_decay

class TextConvNet(ModelsBase):

    def __init__(self, config, batch_size, is_train=True, enable_summary = False):
        ModelsBase.__init__(self)
        self.keep_prop = config.conv_keep_prob
        self.conv2d_layer_count = 1
        self.pred = None
        self.accuracy = None
        self.enable_summary = enable_summary

        # ///////////////////////////////////////////////////////////////////////////
        self.is_trian = is_train
        self.emb_size = config.emb_size
        self.batch_size = batch_size
        self.sent_len = config.sent_len
        self.min_window = config.min_window
        self.max_window = config.max_window
        self.num_kernel = config.num_kernel

        self.num_classes = config.num_class
        self.l2_reg = config.l2_reg


    def build_graph(self):
        """Build the computation graph"""
        self._inputs = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.sent_len], name="conv_input_x")
        self.labels = tf.placeholder(dtype=tf.int32, shape=[self.batch_size], name="conv_target")
        losses = []

        #lookup layer
        with tf.variable_scope("lookup") as scope:
            self._W_emb = _variable_on_cpu("embedding", shape=[self.batch_size, self.sent_len],
                                          initializer=tf.random_uniform_initializer(minval=-1.0, maxval=+1.0))

            # sent_batch is of shape: (batch_size, sent_len, emb_size)
            sent_batch = tf.nn.embedding_lookup(params=self._W_emb, ids=self._inputs)
            # sent_batch is of shape: (batch_size, sent_len, emb_size, 1), in order to use conv2d
            sent_batch = tf.expand_dims(sent_batch, -1)

        with tf.variable_scope("conv") as scope:
            pool_tensors = []
            for kernel_size in range(self.min_window, self.max_window):
                kernel, weight_decay = _variable_with_weight_decay(name='kernel_'+str(kernel_size),
                                                                   shape=[kernel_size, self.emb_size, 1, self.num_kernel],
                                                                   initializer=tf.truncated_normal_initializer(stddev=0.01))
                losses.append(weight_decay)
                conv = tf.nn.conv2d(input=sent_batch, filter=kernel, strides=[1,1,1,], padding='VALID')
                biases = _variable_on_cpu('biases_'+str(kernel_size),
                                          [self.num_kernel],
                                          tf.constant_initializer(0.0))
                bias = tf.nn.bias_add(conv, biases)

                relu = tf.nn.relu(bias, name=scope.name)
                # shape of relu: [batch_size, conv_len, 1, num_kernel]
                conv_len = relu.get_shape()[1]
                pool = tf.nn.max_pool(relu, ksize=[1,conv_len,1,1], strides=[1,1,1,1], padding='VALID')
                # shape of pool: [batch_size, 1, 1, num_kernel]
                pool = tf.squeeze(pool, squeeze_dims=[1,2]) # size: [batch_size, num_kernel]
                pool_tensors.append(pool)
            pool_layer = tf.concat(values=pool_tensors, axis=1, name='pool')

            # drop out layer
            if self.is_train and self.dropout > 0:
                pool_dropout = tf.nn.dropout(pool_layer, 1 - self.dropout)
            else:
                pool_dropout = pool_layer

            # fully-connected layer
            pool_size = (self.max_window - self.min_window + 1) * self.num_kernel # 5 - 3 = 2 + 1 = 3 * 100 = 300
            with tf.variable_scope('fc') as scope:
                W, weight_decay = _variable_with_weight_decay('W', shape=[pool_size, self.num_classes],
                                                              initializer=tf.truncated_normal_initializer(stddev=0.05),
                                                              wd=self.l2_reg)
                losses.append(weight_decay)
                biases = _variable_on_cpu('biases', [self.num_classes], tf.constant_initializer(0.01))
                logits = tf.nn.bias_add(tf.matmul(pool_dropout, W), biases)

            # loss
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self.labels, name="cross_entropy_per_example")
            cross_entropy_loss = tf.reduce_mean(cross_entropy, name='cross_entropy_loss')
            losses.append(cross_entropy)

            self._total_loss = tf.add_n(losses, name='total_loss')

            #correct_prediction
            correct_prediction = tf.to_int32(tf.nn.in_top_k(logits, self.labels, 1))
            self._true_count_op = tf.reduce_sum(correct_prediction)

        # train on a batch
        self._lr = tf.Variable(0.0, trainable=False)
        if self.is_train:
            if self.optimizer == 'adadelta':
                opt = tf.train.AdadeltaOptimizer(self._lr)
            elif self.optimizer == 'adagrad':
                opt = tf.train.AdagradOptimizer(self._lr)
            elif self.optimizer == 'adam':
                opt = tf.train.AdamOptimizer(self._lr)
            elif self.optimizer == 'sgd':
                opt = tf.train.GradientDescentOptimizer(self._lr)
            else:
                raise ValueError("Optimizer not supported.")
            grads = opt.compute_gradients(self._total_loss)
            self._train_op = opt.apply_gradients(grads)

            for var in tf.trainable_variables():
                tf.histogram_summary(var.op.name, var)
        else:
            self._train_op = tf.no_op()

        return

    #///////////////////////////////////////////////////////////////////////////

    def new_weights(self, shape):
        with tf.name_scope("kernel"):
            w = tf.Variable(tf.truncated_normal(shape, stddev=0.05))
        return w

    def new_biases(self, length):
        # equivalent to y intercept
        # constant value carried over across matrix math
        with tf.name_scope("bias"):
            b = tf.Variable(tf.constant(0.05, shape=[length]))
        return b


    def kernel_to_image(self, kernel):
        """

        :param kernel: Convolutional kernel used of shape [height, width, channels, conv_filter_numbers] [0,1,2,3]
        :return: 
        """

        # print("test: ", self.conv2d_layer_count)
        with tf.variable_scope('filter-visualization'):
            # scale weights to [0 1], type is still float
            x_min = tf.reduce_min(kernel)
            x_max = tf.reduce_max(kernel)
            kernel_0_to_1 = (kernel - x_min) / (x_max - x_min)

            # to tf.image_summary format [conv_filter_numbers, height, width, channels]
            kernel_transposed = tf.transpose(kernel_0_to_1, [3, 0, 1, 2])

            shape = kernel_transposed.get_shape()

            if (shape[3].value < 5):
                # this will display random 3 filters from the 64 in conv
                filter_summary = tf.summary.image('conv-layer-' + str(self.conv2d_layer_count) + '/4-channel/filter/',
                                                  kernel_transposed)
                self.train_summaries.append(filter_summary)
            else:
                batch_images = tf.unstack(value=kernel_transposed, axis=3)
                # print("images: ", batch_images)
                for i, batch in enumerate(batch_images):
                    images = tf.unstack(value=batch, axis=0)
                    for image in images:
                        image_reshaped = tf.expand_dims(image, 0)
                        image_reshaped = tf.expand_dims(image_reshaped, -1)
                        filter_summary = tf.summary.image(
                            'conv-layer-' + str(self.conv2d_layer_count) + '/1-channel/filter-' + str(i),
                            image_reshaped)
                        self.train_summaries.append(filter_summary)

    def conv2d_maxpool(self, x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
        """
        Apply convolution then max pooling to x_tensor
        :param x_tensor: TensorFlow Tensor of shape=[batch, in_height, in_width, in_channels]
        :param conv_num_outputs: Number of outputs for the convolutional layer
        :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
        :param conv_strides: Stride 2-D Tuple for convolution
        :param pool_ksize: kernal size 2-D Tuple for pool
        :param pool_strides: Stride 2-D Tuple for pool
        : return: A tensor that represents convolution and max pooling of x_tensor
        """
        with tf.name_scope("conv2d_maxpool_layer"):
            # print("---------------------------------------------------")
            num_input_channel = x_tensor.get_shape()
            assert (len(num_input_channel) == 4)
            shape = [conv_ksize[0], conv_ksize[1], num_input_channel[3].value, conv_num_outputs]
            kernel = self.new_weights(shape=shape)

            biases = self.new_biases(length=conv_num_outputs)
            #
            # print(kernel)
            # print(biases)

            layer = tf.nn.conv2d(input=x_tensor,
                                 filter=kernel,
                                 strides=[1, conv_strides[0], conv_strides[1], 1],
                                 padding='SAME')
            # print("conv2d Layer ===> ", layer)

            layer += biases #tf.nn.bias_add

            # print(layer)

            layer = tf.nn.relu(layer)

            # print("relu Layer ===> ", layer)

            layer = tf.nn.max_pool(value=layer,
                                   ksize=[1, pool_ksize[0], pool_ksize[1], 1],
                                   strides=[1, pool_strides[0], pool_strides[1], 1],
                                   padding='SAME')

            # print(layer)

            #self.kernel_to_image(kernel) Too expensive !!!

        self.conv2d_layer_count += 1
        return layer

    def output(self, x_tensor, num_outputs):
        with tf.name_scope("conv_output_layer"):
            x_shape = x_tensor.get_shape()
            assert (len(x_shape) == 2)
            num_inputs = x_shape[1].value
            # Create new weights and biases.
            weigths = self.new_weights(shape=[num_inputs, num_outputs])
            biases = self.new_biases(length=num_outputs)

            # Calculate the layer as the matrix multiplication of
            # the input and weights, and then add the bias-values.
            # layer = tf.matmul(x_tensor, weights) + biases

            logits = tf.nn.bias_add(tf.matmul(x_tensor, weigths), biases)

        return logits

    def build_graph_for_pretrained_layer(self, batch_size, prev_layer, target):
        """
        Runs Convolutional net work on pretrianed layer
        :param prev_layer: TensorFlow Tensor of shape=[batch_size, hidden_layer_size]
        :param targets: TensorFlow Tensor of shape=[batch_size]
        """
        with tf.name_scope("conv_target"):
            self.labels = target

        conv_ksize = (5, 5)
        conv_strides = (1, 1)
        pool_ksize = (3, 3)
        pool_strides = (3, 3)

        prev_layer_size = prev_layer.get_shape()
        # assert(prev_layer_size[0].value == batch_size)

        layer = self.conv2d_maxpool(prev_layer, 16, conv_ksize, conv_strides, pool_ksize, pool_strides)

        layer = tf.nn.dropout(layer, self.keep_prop)

        layer = self.conv2d_maxpool(layer, 32, conv_ksize, conv_strides, pool_ksize, pool_strides)

        layer = tf.contrib.layers.flatten(layer)#self.flatten(layer)

        layer = tf.nn.dropout(layer, self.keep_prop)

        layer = tf.contrib.layers.fully_connected(layer, 1024, activation_fn=tf.nn.relu) #self.fully_conn(layer, 1024)

        #######################New way: For one class#######################

        # self.pred = tf.identity(tf.contrib.layers.fully_connected(layer, 1, activation_fn=tf.nn.sigmoid), name='conv_pred')
        # self.loss = tf.losses.mean_squared_error(tf.expand_dims(self.labels, 1), self.pred)
        #
        # print(self.pred)
        # print(self.loss)
        # with tf.name_scope("conv_accuracy"):
        #     self.correct_pred = tf.equal(tf.cast(tf.round(self.pred), tf.int32), self.labels)
        #     self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        #
        # if (self.enable_summary): accuracy_summary = tf.summary.scalar("conv-accuracy", self.accuracy)
        # if (self.enable_summary): self.train_summaries.append(accuracy_summary)
        # if (self.enable_summary): self.val_summaries.append(accuracy_summary)

        #################old way################

        self.logits = tf.layers.dense(inputs=layer, units=2, name="conv_logits_layer")
        self.logits_with_softmax = tf.nn.softmax(self.logits, name='conv_logits')

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels,
                                                                       name='cross_entropy_per_example')

        cross_entropy_loss = tf.reduce_mean(cross_entropy, name='cross_entropy_loss')

        self.loss = tf.add_n([cross_entropy_loss], name='total_loss')

        self.predictions = tf.argmax(self.logits, 1, name="conv_pred")

        # Accuracy
        with tf.name_scope("conv_accuracy"):
            correct_predictions = tf.equal(tf.cast(self.predictions, tf.int32), self.labels)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

            if (self.enable_summary): accuracy_summary = tf.summary.scalar("conv-accuracy", self.accuracy)
            if (self.enable_summary): self.train_summaries.append(accuracy_summary)
            if (self.enable_summary): self.val_summaries.append(accuracy_summary)

        # self.pred = tf.argmax(self.logits_with_softmax, 1, name="conv_pred")
        #
        # self._true_count_op = tf.reduce_sum(self.pred)
        #
        # with tf.name_scope("conv_accuracy"):
        #     # correct_prediction
        #     correct_prediction = tf.to_int32(tf.nn.in_top_k(self.logits, self.labels, 1))
        #     self.accuracy = tf.reduce_mean(correct_prediction)
        #     # self.accuracy = tf.reduce_mean(tf.cast(self.pred, tf.float32))
        #
        #     if (self.enable_summary): accuracy_summary = tf.summary.scalar("conv-accuracy", self.accuracy)
        #     if (self.enable_summary): self.train_summaries.append(accuracy_summary)
        #     if (self.enable_summary): self.val_summaries.append(accuracy_summary)

        print('Done Adding Conv Net!')
        return

    @property
    def current_logits(self):
        return self.logits

    @property
    def current_pred(self):
        return self.predictions

    @property
    def current_cross_entropy(self):
        return self.cross_entropy

    @property
    def current_loss(self):
        print("Returing TextConvNet loss")
        return self.loss

    @property
    def current_accuracy(self):
        return self.accuracy