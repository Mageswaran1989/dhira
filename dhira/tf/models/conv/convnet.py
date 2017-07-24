import tensorflow as tf
from overrides import overrides
from dhira.tf.models.base_tf_model import BaseTFModel

class ConvNet(BaseTFModel):
    @overrides
    def __init__(self, config_dict):
        ''

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

    def neural_net_keep_prob_input(self,):
        """
        Return a Tensor for keep probability
        : return: Tensor for keep probability
        """
        ph = tf.placeholder(tf.float32, shape=(None), name="keep_prob")
        return ph


    @overrides
    def _create_placeholders(self):
        ''

    @overrides
    def _build_forward(self):
        ''

    @overrides
    def _get_train_feed_dict(self, batch):
        ''

    @overrides
    def _get_validation_feed_dict(self, batch):
        ''

    @overrides
    def _get_test_feed_dict(self, batch):
        ''
