import logging

import tensorflow as tf
from overrides import overrides

from dhira.tf.models.internal.base_tf_model import BaseTFModel

logger = logging.getLogger(__name__)

class Cifiar10ConvNet(BaseTFModel):
    """
    References: 
        https://www.tensorflow.org/tutorials/deep_cnn
        http://cs231n.github.io/convolutional-networks/
    
    Common Conv Steps:
    1. Function to create a weights
    2. Function to create bias
    3. Apply convolution with appropriate weights and bias to given input layer
    4. Apply max pooling to previous layer
    5. Apply activation function to previous layer
    6. Repeat step 3 to 5 for multiple levels of conv -> maxpool -> relu layers
    7. Flatten the last layer equal to the required output classes
    8. USe appropriate loss and optimization function
    
    Notes:
    SAME padding: 
    - "SAME" tries to pad evenly left and right, but if the amount of columns to be added is odd, 
      it will add the extra column to the right, as is the case in this example 
      (the same logic applies vertically: there may be an extra row of zeros at the bottom).
    - Output size is the same as input size. 
    - This requires the filter window to slip outside input map, hence the need to pad.
    - The output height and width are computed as:
        out_height = ceil(float(in_height) / float(strides[1]))
        out_width = ceil(float(in_width) / float(strides[2]))
        
            "SAME" = with zero padding:
                   pad|                                      |pad
       inputs:      0 |1  2  3  4  5  6  7  8  9  10 11 12 13|0  0
                   |________________|
                                  |_________________|
                                                 |________________|
                                                 
        Input width = 13
        Filter width = 6
        Stride = 5
        
    VALID padding:
    - "VALID" only ever drops the right-most columns (or bottom-most rows).
    - Filter window stays at valid position inside input map.
    - So output size shrinks by filter_size - 1. No padding occurs. 
    - The output height and width are computed as:
        out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
        out_width = ceil(float(in_width - filter_width + 1) / float(strides[2]))
        The easiest case, means no padding at all. Just leave your data the same it was.
        
            "VALID" = without padding:
       inputs:         1  2  3  4  5  6  7  8  9  10 11 (12 13)
                      |________________|                dropped
                                     |_________________|
        Input width = 13
        Filter width = 6
        Stride = 5
        
    Output image size:
     ((W−F+2P)/S) + 1 .
    where:
    W - input volume size
    F - convolution kernel/field/filter size
    P - Number of Zero PAdding
    S - Stride size
    """
    @overrides
    def __init__(self, name, save_dir, log_dir, run_id,
                 image_shape, keep_prop_value):
        super(Cifiar10ConvNet, self).__init__(name=name,
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

    def conv2d_maxpool(self, x_tensor, conv_filter_size, 
                       conv_ksize, conv_strides, 
                       pool_ksize, pool_strides):
        """
        Apply convolution then max pooling to x_tensor
        :param x_tensor: TensorFlow Tensor
        :param conv_filter_size: Number of outputs for the convolutional layer
        :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
        :param conv_strides: Stride 2-D Tuple for convolution
        :param pool_ksize: kernal size 2-D Tuple for pool
        :param pool_strides: Stride 2-D Tuple for pool
        : return: A tensor that represents convolution and max pooling of x_tensor
        """
        logger.info("Creating conv2d_maxpool layer")
        num_input_channel = x_tensor.get_shape()
        assert (len(num_input_channel) == 4)
        #[3,3,4,16]
        shape = [conv_ksize[0], conv_ksize[1], num_input_channel[3].value, conv_filter_size]
        weights = self.new_weights(shape=shape)
        biases = self.new_biases(length=conv_filter_size)

        layer = tf.nn.conv2d(input=x_tensor,
                             filter=weights,
                             strides=[1, conv_strides[0], conv_strides[1], 1],
                             padding='SAME')
        layer += biases
        layer = tf.nn.relu(layer)
        logger.info("   ---> conv layer: {}".format(layer.get_shape()))

        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, pool_ksize[0], pool_ksize[1], 1],
                               strides=[1, pool_strides[0], pool_strides[1], 1],
                               padding='SAME')
        logger.info("   ---> max pool layer: {}".format(layer.get_shape()))

        return layer

    def flatten(self, x_tensor):
        """
        Flatten x_tensor to (Batch Size, Flattened Image Size)
        :param x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
        :param return: A tensor of size (Batch Size, Flattened Image Size).
        """
        layer_shape = x_tensor.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(x_tensor, [-1, num_features])

        return layer_flat

    def fully_connected_with_activation(self, in_tensor, num_outputs):
        """
        Apply a fully connected layer to x_tensor using weight and bias
        :param x_tensor: A 2-D tensor where the first dimension is batch size.
        :param num_outputs: The number of output that the new tensor should be.
        :param return: A 2-D tensor where the second dimension is num_outputs.
        """
        in_shape = in_tensor.get_shape()
        assert (len(in_shape) == 2)
        num_inputs = in_shape[1].value
        # Create new weights and biases.
        weights = self.new_weights(shape=[num_inputs, num_outputs])
        biases = self.new_biases(length=num_outputs)

        # Calculate the layer as the matrix multiplication of
        # the input and weights, and then add the bias-values.
        layer = tf.matmul(in_tensor, weights) + biases
        layer = tf.nn.relu(layer) 
        return layer

    def output(self, in_tensor, num_outputs):
        """
        Apply a output layer to x_tensor using weight and bias
        :param in_tensor: A 2-D tensor where the first dimension is batch size.
        :param num_outputs: The number of output that the new tensor should be.
        :return: A 2-D tensor where the second dimension is num_outputs.
        """
        in_shape = in_tensor.get_shape()
        assert (len(in_shape) == 2)
        num_inputs = in_shape[1].value
        # Create new weights and biases.
        weights = self.new_weights(shape=[num_inputs, num_outputs])
        biases = self.new_biases(length=num_outputs)

        # Calculate the layer as the matrix multiplication of
        # the input and weights, and then add the bias-values.
        layer = tf.matmul(in_tensor, weights) + biases

        return layer

    def conv_net(self, x, keep_prob):
        """
        Create a convolutional neural network model
        :param x: Placeholder tensor that holds image data.
        :param keep_prob: Placeholder tensor that hold dropout keep probability.
        :return: Tensor that represents logits
        """
        shape = x.get_shape()
        assert (len(shape) == 4)

        num_outputs = 10  # number of classes

        logger.info("in_tensor: {}".format(x.get_shape()))
        conv_ksize = (4, 4)
        conv_strides = (1, 1)
        pool_ksize = (2, 2)
        pool_strides = (2, 2)
        layer = self.conv2d_maxpool(x, 16, conv_ksize, conv_strides, pool_ksize, pool_strides)
        logger.info("conv1 layer: {}".format(layer.get_shape()))
        layer = tf.nn.dropout(layer, keep_prob)
        logger.info("conv1_dropout layer: {}".format(layer.get_shape()))
        conv_ksize = (2, 2)
        conv_strides = (1, 1)
        pool_ksize = (2, 2)
        pool_strides = (2, 2)
        layer = self.conv2d_maxpool(layer, 32, conv_ksize, conv_strides, pool_ksize, pool_strides)
        logger.info("conv2 layer: {}".format(layer.get_shape()))

        layer = self.flatten(layer)
        logger.info("flatten layer: {}".format(layer.get_shape()))
        layer = tf.nn.dropout(layer, keep_prob)
        logger.info("flatten_dropout layer : {}".format(layer.get_shape()))

        layer = self.fully_connected_with_activation(layer, 1024)
        logger.info("fully_connected1 layer: {}".format(layer.get_shape()))
        layer = self.fully_connected_with_activation(layer, 512)
        logger.info("fully_connected2 layer: {}".format(layer.get_shape()))

        layer = self.output(layer, num_outputs)
        logger.info("output layer: {}".format(layer.get_shape(), "\n"))

        #  return output
        return layer

    @overrides
    def _setup_graph_def(self):
        # Model
        logits = self.conv_net(self.in_images, self.keep_prop)

        # Name logits Tensor, so that is can be loaded from disk after training
        logits = tf.identity(logits, name='logits')

        self.predictions = logits

        # Loss and Optimizer
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels))
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss, global_step=self.global_step)

        # Accuracy
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(self.labels, 1))
        self.eval_operation = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

        # self._add_scalar_summary(self.loss)
        # self._add_scalar_summary(self.accuracy)

    @overrides
    def _get_train_feed_dict(self, batch):
        images, labels = batch
        feed_dict = {self.in_images: images[0],
                     self.labels: labels[0],
                     self.keep_prop: self.keep_prop_value}
        return feed_dict

    @overrides
    def _get_validation_feed_dict(self, batch):
        images, labels = batch
        feed_dict = {self.in_images: images[0],
                     self.labels: labels[0],
                     self.keep_prop: 1.0}
        return feed_dict

    @overrides
    def _get_test_feed_dict(self, batch):
        images, labels = batch
        feed_dict = {self.in_images: images[0],
                     self.keep_prop: 1.0}
        return feed_dict


# INFO:dhira.tf.models.internal.base_tf_model:Building graph...
# INFO:dhira.tf.models.conv.cifiar_convnet:in_tensor: (?, 32, 32, 3)
# INFO:dhira.tf.models.conv.cifiar_convnet:Creating conv2d_maxpool layer
# INFO:dhira.tf.models.conv.cifiar_convnet:   ---> conv layer: (?, 32, 32, 16)
# INFO:dhira.tf.models.conv.cifiar_convnet:   ---> max pool layer: (?, 16, 16, 16)
# INFO:dhira.tf.models.conv.cifiar_convnet:conv1 layer: (?, 16, 16, 16)
# INFO:dhira.tf.models.conv.cifiar_convnet:conv1_dropout layer: (?, 16, 16, 16)
# INFO:dhira.tf.models.conv.cifiar_convnet:Creating conv2d_maxpool layer
# INFO:dhira.tf.models.conv.cifiar_convnet:   ---> conv layer: (?, 16, 16, 32)
# INFO:dhira.tf.models.conv.cifiar_convnet:   ---> max pool layer: (?, 8, 8, 32)
# INFO:dhira.tf.models.conv.cifiar_convnet:conv2 layer: (?, 8, 8, 32)
# INFO:dhira.tf.models.conv.cifiar_convnet:flatten layer: (?, 2048)
# INFO:dhira.tf.models.conv.cifiar_convnet:flatten_dropout layer : (?, 2048)
# INFO:dhira.tf.models.conv.cifiar_convnet:fully_connected1 layer: (?, 1024)
# INFO:dhira.tf.models.conv.cifiar_convnet:fully_connected2 layer: (?, 512)
# INFO:dhira.tf.models.conv.cifiar_convnet:output layer: (?, 10)