from tensorflow.contrib.tensorboard.plugins import projector
import os
import tensorflow as tf

class ModelsBase(object):

    def __init__(self):

        #To store all summaries of the model
        self.train_summaries = []
        self.val_summaries = []

        self.config_embeddings_filename_tuple = []

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        print(self.dropout_keep_prob)
        self.model_output = None

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
        self.train_summaries.append(train_scalar)

        val_scalar = tf.summary.scalar('scalar/validation/'+name, tensor_obj)
        self.val_summaries.append(val_scalar)


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
        self.train_summaries.append(train_hist)

        val_hist = tf.summary.histogram('hist/validation/'+name, tensor_obj)
        self.val_summaries.append(val_hist)

    def add_embeddings(self, var_name, tsv_file_name):
        config = projector.ProjectorConfig()
        # You can add multiple embeddings. Here we add only one.
        embedding = config.embeddings.add()
        embedding.tensor_name = var_name
        # Link this tensor to its metadata file (e.g. labels).
        self.config_embeddings_filename_tuple.append((config, embedding, tsv_file_name+'.tsv'))

    def build(self):
        raise NotImplementedError("Builds the model and return a tensor output compatible with output classes")

    def loss(self):
        raise NotImplementedError("Return the tensor node that holds the current step/iteration loss")

    def get_predictions(self):
        raise NotImplementedError("Return the tensor node that holds the current step/iteration predictions")

    def accuracy(self):
        raise NotImplementedError("Return the tensor node that holds the current step/iteration accuracy")

    def image_matrix_3_channels(self):
        raise NotImplementedError("Return the tensor that holds the current step/iteration 4-D data that "
                                  "can be reused as input to next Conv model")

    def image_matrix_4_channels(self):
        raise NotImplementedError(".....")

    @property
    def current_train_summaries(self):
        if (len(self.train_summaries) == 0):
            # raise Warning("No summaries added for model of interest :", __class__.__name__) #TODO get dervied calss name
            print("No train summaries added for model of interest :", __class__.__name__)
            return self.train_summaries
        else:
            return self.train_summaries

    @property
    def current_val_summaries(self):
        if (len(self.val_summaries) == 0):
            # raise Warning("No validation summaries added for model of interest :", __class__.__name__)
            print("No validation summaries added for model of interest :", __class__.__name__)
            return self.val_summaries
        else:
            return self.val_summaries