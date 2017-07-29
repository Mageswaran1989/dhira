import tensorflow as tf
import logging
from copy import deepcopy
from overrides import overrides
from dhira.tf.models.base_tf_model import BaseTFModel

logger = logging.getLogger(__name__)

class Cifiar10ConvNet(BaseTFModel):
    """
    TensorFlow Reference: https://www.tensorflow.org/tutorials/deep_cnn
    """
    @overrides
    def __init__(self, name, mode, save_dir, log_dir, run_id,
                 image_shape, keep_prop_value):
        super(Cifiar10ConvNet, self).__init__(name=name,
                                              mode=mode,
                                              run_id=run_id,
                                              save_dir=save_dir,
                                              log_dir=log_dir)
        self.image_shape = image_shape
        self.keep_prop_value = keep_prop_value
        self.num_classes = 10 #from the dataset
        self.in_images = None
        self.labels = None
        self.keep_prop = None

    def neural_net_image_input(self, image_shape, name: str):
        """
        Return a Tensor for a batch of image input
        : image_shape: Shape of the images
        : return: Tensor for image input.
        """
        assert (len(image_shape) == 3)
        ph = tf.placeholder(tf.float32, shape=(None, image_shape[0], image_shape[1], image_shape[2]), name=name)
        return ph

    def neural_net_label_input(self, n_classes, name: str):
        """
        Return a Tensor for a batch of label input
        : n_classes: Number of classes
        : return: Tensor for label input.
        """
        ph = tf.placeholder(tf.float32, shape=(None, n_classes), name=name)
        return ph

    def neural_net_keep_prob_input(self):
        """
        Return a Tensor for keep probability
        : return: Tensor for keep probability
        """
        ph = tf.placeholder(tf.float32, shape=(None), name="keep_prob")
        return ph

    @overrides
    def _create_placeholders(self):
        self.in_images = self.neural_net_image_input(self.image_shape, 'input_image')
        self.labels = self.neural_net_label_input(self.num_classes, 'label')
        self.keep_prop = self.neural_net_keep_prob_input()

    def new_weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    def new_biases(self, length):
        # equivalent to y intercept
        # constant value carried over across matrix math
        return tf.Variable(tf.constant(0.05, shape=[length]))

    def conv2d_maxpool(self, x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
        """
        Apply convolution then max pooling to x_tensor
        :param x_tensor: TensorFlow Tensor
        :param conv_num_outputs: Number of outputs for the convolutional layer
        :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
        :param conv_strides: Stride 2-D Tuple for convolution
        :param pool_ksize: kernal size 2-D Tuple for pool
        :param pool_strides: Stride 2-D Tuple for pool
        : return: A tensor that represents convolution and max pooling of x_tensor
        """
        logger.info("Creating conv2d_maxpool layer")
        num_input_channel = x_tensor.get_shape()
        assert (len(num_input_channel) == 4)
        shape = [conv_ksize[0], conv_ksize[1], num_input_channel[3].value, conv_num_outputs]
        #     print(shape)
        #     print(type(shape))
        weights = self.new_weights(shape=shape)
        biases = self.new_biases(length=conv_num_outputs)

        layer = tf.nn.conv2d(input=x_tensor,
                             filter=weights,
                             strides=[1, conv_strides[0], conv_strides[1], 1],
                             padding='SAME')
        logger.info("   ---> conv layer: ", layer.get_shape())
        layer += biases

        layer = tf.nn.relu(layer)
        logger.info("   ---> relu layer: ", layer.get_shape())

        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, pool_ksize[0], pool_ksize[1], 1],
                               strides=[1, pool_strides[0], pool_strides[1], 1],
                               padding='SAME')
        logger.info("   ---> max pool layer: ", layer.get_shape())

        return layer

    def flatten(self, x_tensor):
        """
        Flatten x_tensor to (Batch Size, Flattened Image Size)
        : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
        : return: A tensor of size (Batch Size, Flattened Image Size).
        """
        layer_shape = x_tensor.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(x_tensor, [-1, num_features])

        return layer_flat

    def fully_conn(self, x_tensor, num_outputs):
        """
        Apply a fully connected layer to x_tensor using weight and bias
        : x_tensor: A 2-D tensor where the first dimension is batch size.
        : num_outputs: The number of output that the new tensor should be.
        : return: A 2-D tensor where the second dimension is num_outputs.
        """
        x_shape = x_tensor.get_shape()
        assert (len(x_shape) == 2)
        num_inputs = x_shape[1].value
        # Create new weights and biases.
        weights = self.new_weights(shape=[num_inputs, num_outputs])
        biases = self.new_biases(length=num_outputs)

        # Calculate the layer as the matrix multiplication of
        # the input and weights, and then add the bias-values.
        layer = tf.matmul(x_tensor, weights) + biases
        layer = tf.nn.relu(layer)  ### TODO should this be softmax?
        return layer

    def output(self, x_tensor, num_outputs):
        """
        Apply a output layer to x_tensor using weight and bias
        : x_tensor: A 2-D tensor where the first dimension is batch size.
        : num_outputs: The number of output that the new tensor should be.
        : return: A 2-D tensor where the second dimension is num_outputs.
        """
        x_shape = x_tensor.get_shape()
        assert (len(x_shape) == 2)
        num_inputs = x_shape[1].value
        # Create new weights and biases.
        weights = self.new_weights(shape=[num_inputs, num_outputs])
        biases = self.new_biases(length=num_outputs)

        # Calculate the layer as the matrix multiplication of
        # the input and weights, and then add the bias-values.
        layer = tf.matmul(x_tensor, weights) + biases

        return layer

    def conv_net(self, x, keep_prob):
        """
        Create a convolutional neural network model
        : x: Placeholder tensor that holds image data.
        : keep_prob: Placeholder tensor that hold dropout keep probability.
        : return: Tensor that represents logits
        """
        shape = x.get_shape()
        assert (len(shape) == 4)
        batch = shape[0].value
        rows = shape[1].value
        cols = shape[2].value
        channel = shape[3].value

        conv_num_outputs = 32
        conv_ksize = (5, 5)
        conv_strides = (1, 1)
        pool_ksize = (2, 2)
        pool_strides = (2, 2)

        num_outputs = 10  # number of classes

        logger.info("x: ", x.get_shape())
        #  Apply 1, 2, or 3 Convolution and Max Pool layers
        #    Play around with different number of outputs, kernel size and stride
        # Function Definition from Above:
        #    conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)
        layer = self.conv2d_maxpool(x, 16, conv_ksize, conv_strides, pool_ksize, pool_strides)
        logger.info("conv1 layer: ", layer.get_shape())
        layer = tf.nn.dropout(layer, keep_prob)
        logger.info("conv1_dropout layer: ", layer.get_shape())
        layer = self.conv2d_maxpool(layer, 32, conv_ksize, conv_strides, pool_ksize, pool_strides)
        logger.info("conv2 layer: ", layer.get_shape())

        #  Apply a Flatten Layer
        # Function Definition from Above:
        #   flatten(x_tensor)
        layer = self.flatten(layer)
        logger.info("flatten layer: ", layer.get_shape())
        layer = tf.nn.dropout(layer, keep_prob)
        logger.info("flatten_dropout layer : ", layer.get_shape())

        #  Apply 1, 2, or 3 Fully Connected Layers
        #    Play around with different number of outputs
        # Function Definition from Above:
        #   fully_conn(x_tensor, num_outputs)
        layer = self.fully_conn(layer, 1024)
        logger.info("fully_connected1 layer: ", layer.get_shape())
        #     layer = fully_conn(layer, 512)
        #     if(DEBUG_FLAG_): print("fully_connected2 layer: ", layer.get_shape())
        #     layer = fully_conn(layer, num_outputs)

        #  Apply an Output Layer
        #    Set this to the number of classes
        # Function Definition from Above:
        #   output(x_tensor, num_outputs)
        layer = self.output(layer, num_outputs)
        logger.info("output layer: ", layer.get_shape(), "\n")

        #  return output
        return layer

    @overrides
    def _build_forward(self):
        # Model
        logits = self.conv_net(self.in_images, self.keep_prop)

        # Name logits Tensor, so that is can be loaded from disk after training
        logits = tf.identity(logits, name='logits')

        # Loss and Optimizer
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels))
        self.training_op = tf.train.AdamOptimizer().minimize(self.loss)

        # Accuracy
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

        self.add_scalar_summary(self.loss)
        self.add_scalar_summary(self.accuracy)

    @overrides
    def _get_train_feed_dict(self, batch):

        images, labels = batch
        #
        # print('=================> Info')
        # print(len(batch))
        # print(len(labels))
        # print(len(images))
        # print(images[0].shape)

        feed_dict = {self.in_images: images,
                     self.labels: labels,
                     self.keep_prop: self.keep_prop_value}
        return feed_dict

    @overrides
    def _get_validation_feed_dict(self, batch):
        images, labels = batch
        feed_dict = {self.in_images: images,
                     self.labels: labels,
                     self.keep_prop: self.keep_prop_value}
        return feed_dict

    @overrides
    def _get_test_feed_dict(self, batch):
        images, labels = batch
        feed_dict = {self.in_images: images,
                     self.labels: labels,
                     self.keep_prop: self.keep_prop_value}
        return feed_dict
