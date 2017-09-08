import tensorflow as tf
from overrides import overrides
import logging
from dhira.tf.models.internal.base_tf_model import BaseTFModel

logger = logging.getLogger(__name__)

class LinearRegresssion(BaseTFModel):
    """
    Linear Regression
    y = mx + c or y = Ax + c
    y [1x1] = A[1 x n] x[n x 1] + [1x1]
    """
    def __init__(self,
                 feature_type,
                 name = 'LinearRegression',
                 run_id = 0):
        super(LinearRegresssion, self).__init__(name=name,
                                                 run_id=run_id,
                                                 save_dir=None,
                                                 log_dir=None)

    def _create_placeholders(self):
        self.x = tf.placeholder(shape=[None, None], dtype=tf.float32, name='x')
        self.labels = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='labels')

    def _setup_graph_def(self):
        A = tf.Variable(tf.random_normal(shape=[1, self.x.get_shape()[1]],
                             stddev=0.01, mean=1, name='weights'))
        bias = tf.Variable(tf.random_normal(shape=[self.x.get_shape()[1]],
                                         stddev=0.1, mean=0, name='bias'))

        self.y = tf.add(tf.matmul(A, self.x), bias)
        #or self.y = A * x + bias

        #L2 Loss
        self._loss = tf.reduce_mean(tf.square(self.y - self.labels))

        self._optimizer = tf.train.AdadeltaOptimizer(0.1).minimize(self._loss,
                                                             global_step=self.global_step)
        self._accuracy = tf.abs(tf.subtract(self.labels, self.y))

    @overrides
    def _get_loss(self):
        """
        Returns model specific loss function
        :return: Model Loss Function
        """
        return self._loss

    @overrides
    def _get_eval_metric(self):
        """
        Returns model specific evaluation metric Tensor Op
        :return: Model Evaluation Metric Function
        """
        return self._accuracy

    @overrides
    def _get_optimizer(self):
        """
        Returns model specific Optimizer Op
        :return: Model Loss Function
        """
        return self._optimizer

    @overrides
    def _evaluate_model_parameters(self, session):
        """
        Override this method to evaluate model specific parameters.
        Eg. result = session.run(some_operation) 
        """
        logger.info('There are no model specific operation evaluation!')

    @overrides
    def _get_prediction(self):
        """
        Returns model specific prediction Tensor Op here
        :return: Model Prediction Operaiton
        """
        return self.y

    @overrides
    def _get_train_feed_dict(self, batch):
        x_values, labels= batch
        return {self.x : x_values, self.labels: labels}

    @overrides
    def _get_val_feed_dict(self, batch):
        x_values, labels= batch
        return {self.x : x_values, self.labels: labels}

    @overrides
    def _get_val_feed_dict(self, batch):
        x_values, labels= batch
        return {self.x : x_values}