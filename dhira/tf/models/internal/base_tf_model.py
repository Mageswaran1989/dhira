import logging
import os
import time
import json
import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.contrib.tensorboard.plugins import projector
from tqdm import tqdm_notebook as tqdm

from dhira.data.data_manager import DataManager

logger = logging.getLogger(__name__)


class BaseTFModel:
    """
    This class is a base model class for Tensorflow that other Tensorflow
    models should inherit from. It defines a unifying API for training and
    prediction.
    
    Consists of following blocks:
    1. Creating the directories for logging and storing the model
    2. Creating the placeholders
    3. Building/compiling the model
    4. Adds loss and accuracy to the logging by default
    5. Train routine
    6. Test routine
    7. Predict method
    """
    def __init__(self,
                 name,
                 run_id,
                 save_dir=None,
                 log_dir=None):
        """
        
        :param name: name of the model
        :param save_dir: To store model parameters. Deafult is '~/.dhira/models/'
        :param log_dir:  To store model logs. Default is '~/.dhira/models/'
        :param run_id: An integer to indicate the curent run
        """

        self._name = name

        #One of [train|predict], to indicate what you want the model to do.
        self._mode = 'train' # default set to 'train'. Use this in user model for train/predict modes

        #To store all summaries of each model
        self._train_summaries = []
        self._val_summaries = []
        self._config_embeddings_filename_tuple = []
        self._train_summary_op = None
        self._val_summary_op = None
        self._train_summary_writer = None
        self._val_summary_writer = None

        self._is_graph_build = False

        #Misc Varaibles
        #Directory setup
        timestamp = str(int(time.time()))

        self._run_id = str(run_id)

        if save_dir is None: save_dir = os.path.expanduser(os.path.join('~', '.dhira', 'models'))
        if log_dir is None:  log_dir = os.path.expanduser(os.path.join('~', '.dhira', 'logs'))
        self._save_dir = os.path.join(save_dir, self._name, self._run_id.zfill(2) + '/')
        self._log_dir = os.path.join(log_dir, self._name, self._run_id.zfill(2) + timestamp + '/')

        self._setup_dir()

        self.global_step = None

        # Following variables needs to be implemented/linked by the user model definition
        # Use self.global_step in optimization for better logging
        self.predictions = None
        self.loss = None
        self.eval_operation = None
        self.optimizer = None #Training optimization
        self.gradient_and_variance = None #Optional

    def _setup_dir(self):
        """
        Setups the directory for storing the models and logs for Tensorboard
        :return: None
        """
        if not os.path.exists(self._save_dir):
            logger.info("save_dir {} does not exist, "
                        "creating it".format(self._save_dir))
            os.makedirs(self._save_dir)

        # Log the run parameters.
        logger.info("Writing logs to {}".format(self._log_dir))

        if not os.path.exists(self._log_dir):
            logger.info("log path {} does not exist, "
                        "creating it".format(self._log_dir))
            os.makedirs(self._log_dir)

    def _log_params(self):
        """
        Gets the variables state of the model and stores them in a json
        :return: None
        """
        params_path = os.path.join(self._log_dir,  self._name + "params.json")
        logger.info("Writing params to {}".format(params_path))

        params = [(str(k),str(v)) for k,v in self.__dict__.items()]

        with open(params_path, 'w') as params_file:
            json.dump(dict(params), params_file, indent=4)

    def _add_scalar_summary(self, tensor_obj):
        """
        Make sure you log only the needed tensors, or else the 
        graph will be executed for all those that are not part of the 
        training also!
        :param tensor_obj: 
        :return: 
        """
        name = tensor_obj.name.replace(':', '_')
        train_scalar = tf.summary.scalar('scalar/train/'+name, tensor_obj)
        self._train_summaries.append(train_scalar)

        val_scalar = tf.summary.scalar('scalar/validation/'+name, tensor_obj)
        self._val_summaries.append(val_scalar)

    def _add_hist_summary(self, tensor_obj):
        """
        Make sure you log only the needed tensors, or else the 
        graph will be executed for all those that are not part of the 
        training also!
        :param tensor_obj: 
        :return: 
        """
        name = tensor_obj.name.replace(':', '_')
        train_hist = tf.summary.histogram('hist/train/'+name, tensor_obj)
        self._train_summaries.append(train_hist)

        val_hist = tf.summary.histogram('hist/validation/'+name, tensor_obj)
        self._val_summaries.append(val_hist)

    def _add_embeddings(self, var_name, tsv_file_name):
        config = projector.ProjectorConfig()
        # You can add multiple embeddings. Here we add only one.
        embedding = config.embeddings.add()
        embedding.tensor_name = var_name
        # Link this tensor to its metadata file (e.g. labels).
        self._config_embeddings_filename_tuple.append((config, embedding, tsv_file_name+'.tsv'))

    def _setup_summaries(self, sess):
        """
        Get scalar, hostogram etc., summaries and merge it for writer 

        Does by creating needed Summary objects and folders
        :return: 
        """
        # Output directory for models and summaries


        print("Writing to {}\n".format(os.path.abspath(self._log_dir)))

        train_summary_dir = os.path.join(self._log_dir, "summaries", "train")
        self._train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        val_summary_dir = os.path.join(self._log_dir, "summaries", "validation")
        self._val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)

        # Model checkpoints
        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        self.checkpoint_dir = os.path.abspath(os.path.join(self._save_dir, "checkpoints/"))

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self._saver = tf.train.Saver(max_to_keep=10)  # Save model after each epoch

        self.train_summary_op = tf.summary.merge(self._train_summaries)
        self.val_summary_op = tf.summary.merge(self._val_summaries)

        print("--------------------------------------------------")
        print("\ntensorboard --logdir  {}".format(os.path.abspath(self._log_dir)))
        print("\ntensorboard --logdir  {} --port 6007".format(os.path.abspath(self.checkpoint_dir)))
        print("--------------------------------------------------")

    def _create_placeholders(self):
        """
        Method for user model to plugin their TF placeholder(s) creation
        :return: None
        """
        raise NotImplementedError

    def _compile(self):
        """
        Method that builds/compiles the model i.e where TF graph definitions are created
        :return: None
        """
        raise NotImplementedError

    def _evaluate_model_parameters(self, session):
        """
        Override this method to evaluate model specific parameters.
        Eg. result = session.run(some_operation) 
        """
        logger.warning('There are no model specific operation evaluation!')

    def compile(self, seed=42):
        """
        Build the graph, ostensibly by setting up the placeholders and then
        creating/compiling the forward pass.
        
        Note: Clear the previous tensorflow graph definitions

        :param seed: int, optional (default=0)
             The graph-level seed to use when building the graph.
        """
        ops.reset_default_graph()
        self._is_graph_build = True
        self._log_params() #Small trick to get all the variables and log them
        # Create the graph object
        with tf.device("/gpu:0"):
            logger.info("Building graph...")
            tf.set_random_seed(seed)
            self.global_step = tf.get_variable(name="global_step",
                                               shape=[],
                                               dtype='int32',
                                               initializer=tf.constant_initializer(0),
                                               trainable=False)
            self._create_placeholders()
            self._compile()

            self._add_scalar_summary(self.loss)
            self._add_scalar_summary(self.eval_operation)

    def _get_train_feed_dict(self, batch):
        """
        Given a train batch from a batch generator,
        return the appropriate feed_dict to pass to the
        model during training.

        :param batch: tuple of NumPy arrays
            A tuple of NumPy arrays containing the data necessary
            to train.
        """
        raise NotImplementedError

    def _get_validation_feed_dict(self, batch):
        """
        Given a validation batch from a batch generator,
        return the appropriate feed_dict to pass to the
        model during validation.

        :param batch: tuple of NumPy arrays
            A tuple of NumPy arrays containing the data necessary
            to validate.
        """
        raise NotImplementedError

    def _get_test_feed_dict(self, batch):
        """
        Given a test batch from a batch generator,
        return the appropriate feed_dict to pass to the
        model during prediction.

        :param batch: tuple of NumPy arrays
            A tuple of NumPy arrays containing the data necessary
            to predict.
        """
        raise NotImplementedError

    def _evaluate_on_validation(self, get_val_feature_generator,
                                batch_size,
                                num_val_steps,
                                session):
        val_batch_gen = DataManager.get_batch_generator(
            get_val_feature_generator, batch_size)
        # Calculate the mean of the validation metrics
        # over the validation set.
        val_accuracies = []
        val_losses = []
        for val_batch in tqdm(val_batch_gen,
                              total=num_val_steps,
                              desc="Validation Batches Completed",
                              leave=False):

            #Ignore the last batch if the size doesn't match
            # if(len(val_batch) != batch_size):
            #     continue
            feed_dict = self._get_validation_feed_dict(val_batch)
            val_batch_acc, val_batch_loss = session.run(
                [self.eval_operation, self.loss],
                feed_dict=feed_dict)

            val_accuracies.append(val_batch_acc)
            val_losses.append(val_batch_loss)

        # Take the mean of the accuracies and losses.
        # TODO/FIXME this assumes each batch is same shape, which
        # is not necessarily true.
        mean_val_accuracy = np.mean(val_accuracies)
        mean_val_loss = np.mean(val_losses)

        # Create a new Summary object with mean_val accuracy
        # and mean_val_loss and add it to Tensorboard.
        val_summary = tf.Summary(value=[
            tf.Summary.Value(tag="_val_summaries/loss",
                             simple_value=mean_val_loss),
            tf.Summary.Value(tag="_val_summaries/accuracy",
                             simple_value=mean_val_accuracy)])
        return mean_val_accuracy, mean_val_loss, val_summary

    def train(self,
              get_train_feature_generator,
              get_val_feature_generator,
              batch_size,
              num_epochs,
              num_train_steps_per_epoch,
              num_val_steps,
              val_period,
              log_period,
              save_period,
              max_ckpts_to_keep=10,
              patience=0):
        """
        Train the model.

        :param get_train_instance_generator: Function returning generator
            This function should return a finite generator that produces
            features for use in training.

        :param get_val_feature_generator: Function returning generator
            This function should return a finite generator that produces
            features for use in validation.

        :param batch_size: int
            The number of features per batch produced by the generator.

        :param num_train_steps_per_epoch: int
            The number of training steps after which an epoch has passed.

        :param num_epochs: int
            The number of epochs to train for.

        :param num_val_steps: int
            The number of batches generated by the validation batch generator.

        :param save_path: str
            The input path to the tensorflow Saver responsible for
            checkpointing.

        :param log_path: str
            The input path to the tensorflow SummaryWriter responsible for
            logging the progress.

        :param val_period: int, optional (default=250)
            Number of steps between each evaluation of performance on the
            held-out validation set.

        :param log_period: int, optional (default=10)
            Number of steps between each summary op evaluation.

        :param save_period: int, optional (default=250)
            Number of steps between each model checkpoint.

        :param max_ckpts_to_keep: int, optional (default=10)
            The maximum number of model to checkpoints to keep.

        :param patience: int, optional (default=0)
            The number of epochs with no improvement in validation loss
            after which training will be stopped.
        """
        previous_mode = self._mode
        self._mode = 'train'
        if self.predictions is None or \
                        self.loss is None or \
                        self.eval_operation is None or \
                        self.optimizer is None:
            logger.info(self.predictions)
            logger.info(self.loss)
            logger.info(self.predictions)
            logger.info(self.optimizer)
            raise RuntimeError('User model missed to link predictions/loss/eval/optimizer operations!')

        global_step = 0
        init_op = tf.global_variables_initializer()

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=True)
        with tf.Session(config=sess_config) as sess:
            sess.run(init_op)
            
            self._setup_summaries(sess=sess)

            epoch_validation_losses = []
            # Iterate over a generator that returns batches.
            for epoch in tqdm(range(num_epochs), desc="Epochs Completed"):
                # Get a generator of train batches
                train_batch_gen = DataManager.get_batch_generator(
                    get_train_feature_generator, batch_size)
                # Iterate over the generated batches
                for train_batch in tqdm(train_batch_gen,
                                        total=num_train_steps_per_epoch,
                                        desc="Train Batches Completed",
                                        leave=False):

                    global_step = sess.run(self.global_step) + 1

                    feed_dict = self._get_train_feed_dict(train_batch)

                    # Do a gradient update, and log results to Tensorboard
                    # if necessary.
                    if global_step % log_period == 0:
                        # Record summary with gradient update
                        train_loss, _, train_summary = sess.run(
                            [self.loss, self.optimizer, self.train_summary_op],
                            feed_dict=feed_dict)
                        self._train_summary_writer.add_summary(train_summary, global_step)
                    else:
                        # Do a gradient update without recording anything.
                        train_loss, _ = sess.run(
                            [self.loss, self.optimizer],
                            feed_dict=feed_dict)

                    if global_step % val_period == 0:
                        # Evaluate on validation data
                        val_acc, val_loss, val_summary = self._evaluate_on_validation(
                            get_val_feature_generator=get_val_feature_generator,
                            batch_size=batch_size,
                            num_val_steps=num_val_steps,
                            session=sess)
                        self._val_summary_writer.add_summary(val_summary, global_step)
                    # Write a model checkpoint if necessary.
                    if global_step % save_period == 0:
                        ret = self._saver.save(sess, self._save_dir + '/' + self._name, global_step=global_step)
                        logger.info('Saving final model @ ' + os.path.abspath(ret))


                # End of the epoch, so save the model and check validation loss,
                # stopping if applicable.
                model_path = self._saver.save(sess, self._save_dir + '/' + self._name, global_step=global_step)
                logger.info('Saving final model @ ' + os.path.abspath(model_path))

                val_acc, val_loss, val_summary = self._evaluate_on_validation(
                    get_val_feature_generator=get_val_feature_generator,
                    batch_size=batch_size,
                    num_val_steps=num_val_steps,
                    session=sess)
                self._val_summary_writer.add_summary(val_summary, global_step)

                epoch_validation_losses.append(val_loss)

                # Get the lowest validation loss, with regards to the patience
                # threshold.
                patience_val_losses = epoch_validation_losses[:-(patience + 1)]
                if patience_val_losses:
                    min_patience_val_loss = min(patience_val_losses)
                else:
                    min_patience_val_loss = math.inf
                if min_patience_val_loss <= val_loss:
                    # past loss was lower, so stop
                    logger.info("Validation loss of {} in last {} "
                                "epochs, which is lower than current "
                                "epoch validation loss of {}; stopping "
                                "early.".format(min_patience_val_loss,
                                                patience,
                                                val_loss))
                    break

            #Evaluate model specific evaluations
            self._evaluate_model_parameters(sess)

        # Done training!
        logger.info("Finished {} epochs!".format(epoch + 1))
        self._mode = previous_mode
        return os.path.abspath(model_path)

    def test(self,
                get_test_instance_generator,
                model_load_dir,
                batch_size,
                num_test_steps=None):
        """
        Load a serialized model and use it for prediction on a test
        set (from a finite generator).

        :param get_test_instance_generator: Function returning generator
            This function should return a finite generator that produces features
            for use in training.

        :param model_load_dir: str
            Path to a directory with serialized tensorflow checkpoints for the
            model to be run. The most recent checkpoint will be loaded and used
            for prediction.

        :param batch_size: int
            The number of features per batch produced by the generator.

        :param num_test_steps: int
            The number of steps (calculated by ceil(total # test examples / batch_size))
            in testing. This does not have any effect on how much of the test data
            is read; inference keeps going until the generator is exhausted. It
            is used to set a total for the progress bar.
        """
        previous_mode = self._mode
        self._mode = 'predict'

        if num_test_steps is None:
            logger.info("num_test_steps is not set, pass in a value "
                        "to show a progress bar.")

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=True)
        with tf.Session(config=sess_config) as sess:
            saver = tf.train.Saver()
            logger.info("Getting latest checkpoint in {}".format(model_load_dir))
            last_checkpoint = tf.train.latest_checkpoint(model_load_dir)
            logger.info("Attempting to load checkpoint at {}".format(last_checkpoint))
            saver.restore(sess, last_checkpoint)
            logger.info("Successfully loaded {}!".format(last_checkpoint))

            # Get a generator of test batches
            test_batch_gen = DataManager.get_batch_generator(
                get_test_instance_generator, batch_size)

            y_pred = []
            for batch in tqdm(test_batch_gen,
                              total=num_test_steps,
                              desc="Test Batches Completed"):
                feed_dict = self._get_test_feed_dict(batch)
                y_pred_batch = sess.run(self.predictions, feed_dict=feed_dict)
                y_pred.append(y_pred_batch)
            y_pred_flat = np.concatenate(y_pred, axis=0)

        self._mode = previous_mode
        return y_pred_flat

    def predict(self, batched_features, model_load_dir=None):
        """
        Load a serialized model and use it for prediction on user custom inputs.
        
        Use dataset.custom_input() to generate feature(s) and use datamanager.to_batch()
        to prepare the data for prediction.
        :param model_load_dir: str
            Path to a directory with serialized tensorflow checkpoints for the
            model to be run. The most recent checkpoint will be loaded and used
            for prediction.
            Bt default the model will be loaded from ~/.dhira.models/model_name/run_id/
        """

        previous_mode = self._mode
        self._mode = 'predict'

        if model_load_dir is None:
            model_load_dir = self._save_dir
            logger.info('Model is lodded from {}'.format(model_load_dir))

        if not self._is_graph_build:
            logger.info('Initializing the model for prediction...')
            self.compile()

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=True)
        with tf.Session(config=sess_config) as sess:
            saver = tf.train.Saver()
            logger.info("Getting latest checkpoint in {}".format(model_load_dir))
            last_checkpoint = tf.train.latest_checkpoint(model_load_dir)
            logger.info("Attempting to load checkpoint at {}".format(last_checkpoint))
            saver.restore(sess, last_checkpoint)
            logger.info("Successfully loaded {}!".format(last_checkpoint))

            feed_dict = self._get_test_feed_dict(batched_features)
            y_pred = sess.run(self.predictions, feed_dict=feed_dict)

        self._mode = previous_mode
        return y_pred
