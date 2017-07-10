import pickle
import tensorflow as tf
import os
import time
import datetime

from tqdm import tqdm

import numpy as np

from dhira.embeddings_loader import EmbeddingLoader
from dhira.input_preprocessor import QuoraInputProcessor, Tokenizer
from dhira.siamese_network import SiameseLSTM
from random import random

import dhira.global_config
from dhira.text_convnet import *
from tensorflow.contrib.tensorboard.plugins import projector

class Train(object):

    def __init__(self, num_epocs, batch_size, preprocess=False, config=dhira.global_config.Config.FLAGS):
        self.num_epocs = num_epocs
        self.batch_size = batch_size
        self.preprocess_enabled = preprocess
        self.config = config
        self.learning_rate = None

        self.models = []
        self.loss = None
        self.accuracy = None

        self.train_iter_count = 0
        self.val_iter_count = 0
        self.train_step_count = 0
        self.train_loss = 0.0
        self.mean_train_loss = 0.0
        self.val_loss = 0.0
        self.mean_val_loss = 0.0

        self.epoch_loss = 0.0
        self.total_steps = 0

        self.embedding_matrix = None
        self.global_train_step = None
        self.global_val_step = None
        self.global_val_step_op = None
        self.siamese_model = None
        self.conv_model = None
        self.gradient_and_variance = None
        self.train_op = None
        self.train_summary_op = None
        self.val_summary_op = None
        self.train_summary_writer = None
        self.val_summary_writer = None
        self.log_dir = None
        self.checkpoint_dir = None
        self.saver = None
        self.sess = None

        self.dataset = QuoraInputProcessor()
        pass

    # -----------------------------------------------------------------------------------------

    def load_embedding_matrix(self):
        embeddings = EmbeddingLoader()
        self.embedding_matrix = embeddings.load_embeddings()

    # -----------------------------------------------------------------------------------------

    def get_train_batches(self):
        return self.dataset.train_batch_iter(self.batch_size)

    # -----------------------------------------------------------------------------------------

    def get_validation_batches(self):
        return self.dataset.val_batch_iter(self.batch_size)

    # -----------------------------------------------------------------------------------------

    def build_models(self):
        raise NotImplementedError("Needs to be implemented for the specific dataset")

    #-----------------------------------------------------------------------------------------

    def set_optimizer(self):

        with tf.name_scope("Optimizer"):
            # self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")

            self.learning_rate = tf.train.exponential_decay(self.config.lr, self.train_step_count,
                                                       1000, 0.96, staircase=True)

            if self.config.optimizer == 'adadelta':
                self.optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
            elif self.config.optimizer == 'adagrad':
                self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            elif self.config.optimizer == 'adam':
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.4)
            elif self.config.optimizer == 'sgd':
                self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            else:
                raise ValueError("Optimizer not supported.")


            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.gradient_and_variance = self.optimizer.compute_gradients(self.loss)
                self.train_op = self.optimizer.apply_gradients(self.gradient_and_variance,
                                                               global_step=self.global_train_step)

    # -----------------------------------------------------------------------------------------

    def setup_summaries(self):
        """
        Get the summaries from all odels and merge it locally
        
        Does by creating needed Summary objects and folders
        :return: 
        """
        # Output directory for models and summaries
        train_summaries = []
        val_summaries = []

        timestamp = str(int(time.time()))
        self.log_dir = os.path.abspath(os.path.join(os.path.curdir, "logs/", timestamp))
        print("Writing to {}\n".format(self.log_dir))

        # Setup global train summaries
        # train_loss_summary = tf.summary.scalar("loss/scalar/train/"+self.loss.name.replace(':', '_'), self.loss)
        # train_summaries.append(train_loss_summary)
        #
        # val_loss_summary = tf.summary.scalar("loss/scalar/val/"+self.loss.name.replace(':', '_'), self.loss)
        # val_summaries.append(val_loss_summary)

        train_summaries.append(tf.summary.scalar('learning_rate/train', self.learning_rate))
        val_summaries.append(tf.summary.scalar('learning_rate/val', self.learning_rate))

        with tf.name_scope('gradients'):
            for g, v in self.gradient_and_variance:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name.replace(':', '_')), g)
                    grad_scalar_summary = tf.summary.scalar("{}/grad/scalar".format(v.name.replace(':', '_')), tf.nn.zero_fraction(g))
                    train_summaries.append(grad_hist_summary)
                    train_summaries.append(grad_scalar_summary)

        # Now log the summaries from the models
        for model in self.models:
            train_summaries += model.current_train_summaries
            val_summaries += model.current_val_summaries

        self.train_summary_op = tf.summary.merge(train_summaries)
        self.val_summary_op = tf.summary.merge(val_summaries)

        train_summary_dir = os.path.join(self.log_dir, "summaries", "train")
        self.train_summary_writer = tf.summary.FileWriter(train_summary_dir, self.sess.graph)

        val_summary_dir = os.path.join(self.log_dir, "summaries", "validation")
        self.val_summary_writer = tf.summary.FileWriter(val_summary_dir, self.sess.graph)


        #Model checkpoints
        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        self.checkpoint_dir = os.path.abspath(os.path.join(self.log_dir, "checkpoints/"))

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.saver = tf.train.Saver(max_to_keep=1000) # Save model after each epoch

        #Get the embedding matrix if any from the models
        # Use the same LOG_DIR where you stored your checkpoint.
        embedding_writer = tf.summary.FileWriter(self.checkpoint_dir)
        for model in self.models:
            for config, embedding, meta_file_name in model.config_embeddings_filename_tuple:
                path = os.path.join(self.checkpoint_dir, meta_file_name)
                Tokenizer.word_count_to_tsv(tsv_file_path=path) # TODO decouple this one
                embedding.metadata_path = path
                projector.visualize_embeddings(embedding_writer, config)

        print("--------------------------------------------------")
        print("\ntensorboard --logdir  {}\n".format(self.log_dir))
        print("\ntensorboard --logdir  {} --port 6007\n".format(self.checkpoint_dir))
        print("--------------------------------------------------")

    # -----------------------------------------------------------------------------------------

    def build_graph(self):
        """Initializes teh tf session object"""
        print("Starting graph definition")
        with tf.Graph().as_default(), tf.device("/gpu:0"):
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
            with self.sess.as_default():
                # Initialize all variables
                print("Build Models")
                self.build_models()
                print("Setting-up optimizer")
                self.set_optimizer()
                print("Initialize Global Variables")
                self.sess.run(tf.global_variables_initializer())

                print("Setting up summaries and saving TF graph(nodes & edges) for Tensorboard!")
                self.setup_summaries()

    # -----------------------------------------------------------------------------------------

    def step(self, evaluate=False):
        """
        A single training step
        """
        raise NotImplementedError("Needs to be implemented for the sppecific dataset")

    # -----------------------------------------------------------------------------------------

    def reload_batches(self):
        self.train_batches = self.get_train_batches()
        self.validate_batches = self.get_validation_batches()

    # -----------------------------------------------------------------------------------------

    def setup(self):


        #Load data sets
        self.load_embedding_matrix()
        self.train_batches = self.get_train_batches()
        self.validate_batches = self.get_validation_batches()

        # Setup the Tensorflow environment
        self.build_graph()

        # load checkpoint model from previous runs
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

        # self.saver.restore(self.sess, '/opt/deeplearning/git/quorakaggle/logs/1494072095/checkpoints/dhira-model.ckpt-3555')

    def train(self):

        for epoch in range(self.num_epocs):
            from pathlib import Path

            f = Path('__stop__') # Create a file calles '__stop__' to stop the execution on new epoch
            if(f.exists()):
                print('User intetrrupted the training... stoping with run of epoch {}'.format(epoch))
                break # Stop gracefully during execution

            stop = 0
            print('Starting epoch: {} at {}'.format(epoch, datetime.datetime.now().isoformat()))
            self.train_iter_count = epoch + 1

            for batch in tqdm(self.train_batches):
                self.current_train_batch = batch #This will be used in step function
                self.step()
                stop += 1
                # if(stop >5): break

            for batch in tqdm(self.validate_batches):
                self.current_validate_batch = batch#This will be used in step function
                self.step(evaluate=True)
                stop += 1
                # if (stop > 10): break

            #Saving the model after each epoch
            checkpoint_prefix = os.path.join(self.checkpoint_dir, "dhira-model.ckpt")
            self.saver.save(self.sess, checkpoint_prefix, global_step=self.train_step_count)
            print("Model checkpoint saved at: ", checkpoint_prefix+'-'+str(self.train_step_count))

            #TODO is it possible to optimize?
            self.reload_batches()
        return self.checkpoint_dir


class TrainQuoraDataset(Train):
    def __init__(self, num_epocs, batch_size, preprocess=False, config=dhira.global_config.Config.FLAGS):
        Train.__init__(self, num_epocs=num_epocs, batch_size=batch_size, preprocess=preprocess, config=config)
        if (self.preprocess_enabled):
            self.dataset.preprocess()

    def build_models(self):
        """
        1. Initialize the different models
        2. Link the models if needed
        3. Get the loss from the model of interest and store the reference locally
        :return: 
        """
        self.global_train_step = tf.Variable(0, name="global_train_step", trainable=False, dtype=tf.int64)
        self.global_val_step = tf.Variable(0, name="global_val_step", trainable=False, dtype=tf.int64)
        self.global_val_step_op = tf.assign_add(self.global_val_step, 1, name='global_val_step_increment')


        self.siamese_model = SiameseLSTM(self.config, vocab_size=len(self.embedding_matrix), batch_size=self.batch_size, enable_summary=True)
        self.models.append(self.siamese_model)

        # with tf.name_scope("ConvolutionalNetwork"):
        #     self.conv_model = TextConvNet(self.config, self.batch_size, True, enable_summary=True)
        #     self.conv_model.build_graph_for_pretrained_layer(batch_size=self.batch_size,
        #                                                  prev_layer=self.siamese_model.image_matrix_4_channels, #image_matrix_3_channels,
        #                                                  target=self.siamese_model.input_y)
        # self.models.append(self.conv_model)

        self.loss = self.models[-1].loss()
        self.accuracy = self.models[-1].accuracy()

    def step(self, evaluate=False):
        """
        A single training step
        """
        if evaluate:
            dkpb = 1.0
            x1_batch, x2_batch, y_batch = zip(*self.current_validate_batch)
        else:
            dkpb = self.config.default_keep_prob
            x1_batch, x2_batch, y_batch = zip(*self.current_train_batch)

        #TODO : dynamic learning rate




        if random() > 0.5:
            feed_dict = {
                self.siamese_model.input_x1: np.vstack((x1_batch, x2_batch)),
                self.siamese_model.input_x2: np.vstack((x2_batch, x1_batch)),
                self.siamese_model.input_y: np.concatenate((y_batch, y_batch)),
                self.siamese_model.embedding_placeholder: self.embedding_matrix,
                self.siamese_model.dropout_keep_prob: dkpb,
                # self.learning_rate: self.config.lr,
                self.siamese_model.is_training: True
            }
        else:
            feed_dict = {
                self.siamese_model.input_x1: np.vstack((x2_batch, x1_batch)),
                self.siamese_model.input_x2: np.vstack((x1_batch, x2_batch)),
                self.siamese_model.input_y: np.concatenate((y_batch, y_batch)),
                self.siamese_model.embedding_placeholder: self.embedding_matrix,
                self.siamese_model.dropout_keep_prob: dkpb,
                # self.learning_rate: self.config.lr,
                self.siamese_model.is_training: True
            }

        if evaluate:
            feed_dict[self.siamese_model.is_training] = False #disable batch normalization
            _, val_step, step_loss, step_acc, validation_summary = self.sess.run([self.global_val_step_op, self.global_val_step, self.loss, self.accuracy, self.val_summary_op], feed_dict)
            self.val_summary_writer.add_summary(validation_summary, val_step)

            # For local console evaluation
            self.val_iter_count = val_step
            self.val_loss += step_loss
            self.mean_val_loss = self.val_loss / (self.val_iter_count)

            print("Validation epoch/step : ", self.train_iter_count, "/", self.val_iter_count,
                  " Loss : ", step_loss, " Total Loss",   self.mean_val_loss,
                  "Current Accuracy: ", step_acc)
        else:
            _, train_step, step_loss, step_acc, train_summary = self.sess.run([self.train_op, self.global_train_step, self.loss, self.accuracy, self.train_summary_op], feed_dict)
            self.train_summary_writer.add_summary(train_summary, train_step)

            # For local console evaluation
            self.train_step_count = train_step
            self.train_loss += step_loss
            self.mean_train_loss = self.train_loss / (self.train_step_count)

            print("Train epoch/step : ", self.train_iter_count, "/", self.train_step_count,
                  " Loss : ", step_loss, " Total Loss", self.mean_train_loss,
                  "Current Accuracy: ", step_acc)

        #Just to make sure we get a new copies from next batch
        self.current_validate_batch = self.current_train_batch = None