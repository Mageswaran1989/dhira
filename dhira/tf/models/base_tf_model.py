import logging
import os
import time
import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tqdm import tqdm

from dhira.data.data_manager import DataManager

logger = logging.getLogger(__name__)


class BaseTFModel:
    """
    This class is a base model class for Tensorflow that other Tensorflow
    models should inherit from. It defines a unifying API for training and
    prediction.

    :param mode: str
        One of [train|predict], to indicate what you want the model to do.
    """
    def __init__(self, mode):
        self.mode = mode
        self.global_step = tf.get_variable(name="global_step",
                                           shape=[],
                                           dtype='int32',
                                           initializer=tf.constant_initializer(0),
                                           trainable=False)

        # Outputs from the model, defined by child classes
        self.y_pred = None
        self.loss = None
        self.accuracy = None

        self.training_op = None
        self.gradient_and_variance = None

        #To store all summaries of each model
        self.summary_op = None #Needs to be deprecated
        self._train_summaries = []
        self._val_summaries = []
        self._config_embeddings_filename_tuple = []
        self._train_summary_op = None
        self._val_summary_op = None
        self._train_summary_writer = None
        self._val_summary_writer = None

        #Misc Varaibles
        timestamp = str(int(time.time()))
        self._log_dir = os.path.abspath(os.path.join(os.path.curdir, "logs/", timestamp))

    def _create_placeholders(self):
        raise NotImplementedError

    def _build_forward(self):
        raise NotImplementedError

    def build_graph(self, seed=0):
        """
        Build the graph, ostensibly by setting up the placeholders and then
        creating the forward pass.

        :param seed: int, optional (default=0)
             The graph-level seed to use when building the graph.
        """
        logger.info("Building graph...")
        tf.set_random_seed(seed)
        self._create_placeholders()
        self._build_forward()

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

    def add_scalar_summary(self, tensor_obj):
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


    def add_hist_summary(self, tensor_obj):
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

    def add_embeddings(self, var_name, tsv_file_name):
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


        print("Writing to {}\n".format(self._log_dir))

        if (self.gradient_and_variance):
            with tf.name_scope('gradients'):
                for g, v in self.gradient_and_variance:
                    if g is not None:
                        grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name.replace(':', '_')), g)
                        grad_scalar_summary = tf.summary.scalar("{}/grad/scalar".format(v.name.replace(':', '_')),
                                                                tf.nn.zero_fraction(g))
                        self._train_summaries.append(grad_hist_summary)
                        self._train_summaries.append(grad_scalar_summary)

        # Now log the summaries for current model
        # self._train_summary_op = tf.summary.merge(self._train_summaries)
        # self._val_summary_op = tf.summary.merge(self._val_summaries)

        train_summary_dir = os.path.join(self._log_dir, "summaries", "train")
        self._train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        val_summary_dir = os.path.join(self._log_dir, "summaries", "validation")
        self._val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)

        # Model checkpoints
        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        self.checkpoint_dir = os.path.abspath(os.path.join(self._log_dir, "checkpoints/"))

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self._saver = tf.train.Saver(max_to_keep=1000)  # Save model after each epoch

        # # Get the embedding matrix if any from the models
        # # Use the same LOG_DIR where you stored your checkpoint.
        # embedding_writer = tf.summary.FileWriter(self.checkpoint_dir)
        # for model in self.models:
        #     for config, embedding, meta_file_name in model._config_embeddings_filename_tuple:
        #         path = os.path.join(self.checkpoint_dir, meta_file_name)
        #         Tokenizer.word_count_to_tsv(tsv_file_path=path)  # TODO decouple this one
        #         embedding.metadata_path = path
        #         projector.visualize_embeddings(embedding_writer, config)

        print("--------------------------------------------------")
        print("\ntensorboard --logdir  {}\n".format(self._log_dir))
        print("\ntensorboard --logdir  {} --port 6007\n".format(self.checkpoint_dir))
        print("--------------------------------------------------")

    def train(self,
              get_train_instance_generator, get_val_instance_generator,
              batch_size, num_train_steps_per_epoch, num_epochs,
              num_val_steps, save_path, log_path,
              val_period=250, log_period=10, save_period=250,
              max_ckpts_to_keep=10, patience=0):
        """
        Train the model.

        :param get_train_instance_generator: Function returning generator
            This function should return a finite generator that produces
            features for use in training.

        :param get_val_instance_generator: Function returning generator
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

        global_step = 0
        init_op = tf.global_variables_initializer()

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(gpu_options=gpu_options)
        with tf.Session(config=sess_config) as sess:
            sess.run(init_op)
            
            self._setup_summaries(sess=sess)

            epoch_validation_losses = []
            # Iterate over a generator that returns batches.
            for epoch in tqdm(range(num_epochs), desc="Epochs Completed"):
                # Get a generator of train batches
                train_batch_gen = DataManager.get_batch_generator(
                    get_train_instance_generator, batch_size)
                # Iterate over the generated batches
                for train_batch in tqdm(train_batch_gen,
                                        total=num_train_steps_per_epoch,
                                        desc="Train Batches Completed",
                                        leave=False):
                    global_step = sess.run(self.global_step) + 1

                    inputs, targets = train_batch
                    feed_dict = self._get_train_feed_dict(train_batch)

                    # Do a gradient update, and log results to Tensorboard
                    # if necessary.
                    if global_step % log_period == 0:
                        # Record summary with gradient update
                        train_loss, _, train_summary = sess.run(
                            [self.loss, self.training_op, self.summary_op],
                            feed_dict=feed_dict)
                        self._train_summary_writer.add_summary(train_summary, global_step)
                    else:
                        # Do a gradient update without recording anything.
                        train_loss, _ = sess.run(
                            [self.loss, self.training_op],
                            feed_dict=feed_dict)

                    if global_step % val_period == 0:
                        # Evaluate on validation data
                        val_acc, val_loss, val_summary = self._evaluate_on_validation(
                            get_val_instance_generator=get_val_instance_generator,
                            batch_size=batch_size,
                            num_val_steps=num_val_steps,
                            session=sess)
                        self._val_summary_writer.add_summary(val_summary, global_step)
                    # Write a model checkpoint if necessary.
                    if global_step % save_period == 0:
                        self._saver.save(sess, save_path, global_step=global_step)

                # End of the epoch, so save the model and check validation loss,
                # stopping if applicable.
                self._saver.save(sess, save_path, global_step=global_step)
                val_acc, val_loss, val_summary = self._evaluate_on_validation(
                    get_val_instance_generator=get_val_instance_generator,
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

        # Done training!
        logger.info("Finished {} epochs!".format(epoch + 1))

    def predict(self, get_test_instance_generator, model_load_dir, batch_size,
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
        if num_test_steps is None:
            logger.info("num_test_steps is not set, pass in a value "
                        "to show a progress bar.")

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(gpu_options=gpu_options)
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
                y_pred_batch = sess.run(self.y_pred, feed_dict=feed_dict)
                y_pred.append(y_pred_batch)
            y_pred_flat = np.concatenate(y_pred, axis=0)
        return y_pred_flat

    def _evaluate_on_validation(self, get_val_instance_generator,
                                batch_size,
                                num_val_steps,
                                session):
        val_batch_gen = DataManager.get_batch_generator(
            get_val_instance_generator, batch_size)
        # Calculate the mean of the validation metrics
        # over the validation set.
        val_accuracies = []
        val_losses = []
        for val_batch in tqdm(val_batch_gen,
                              total=num_val_steps,
                              desc="Validation Batches Completed",
                              leave=False):
            feed_dict = self._get_validation_feed_dict(val_batch)
            val_batch_acc, val_batch_loss = session.run(
                [self.accuracy, self.loss],
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

