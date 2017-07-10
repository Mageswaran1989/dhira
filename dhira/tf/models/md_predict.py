import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm

from dhira.embeddings_loader import EmbeddingLoader
from dhira.input_preprocessor import QuoraInputProcessor
from dhira.siamese_network import SiameseLSTM
from dhira.global_config import Config

def get_latest_model(checkpoint_dir):
    with open(checkpoint_dir + '/checkpoint') as file:
        lines = file.readlines()
        latest_checkpoint_file = lines[0].split(':')[1].replace('"', '').replace('\n', '').replace(' ', '')
        return latest_checkpoint_file

flatten = lambda l: [item for sublist in l for item in sublist]

class Predict(object):
    def __init__(self, check_point_dir, embedding_matrix_path='embedding_matrix.p', batch_size=128):
        self.checkpoint_file = get_latest_model(check_point_dir)
        self.csv_file_name = self.checkpoint_file.split('/')[-1]+'.csv'
        self.embedding_matrix = EmbeddingLoader().load_embeddings(embedding_matrix_path)
        self.batch_size = batch_size
        self.graph = None
        self.sess = None
        self.saver = None
        self.logits = None
        self.predictions = None
        self.dataset = QuoraInputProcessor()
        self.test_batches = self.dataset.test_batch_iter(batch_size)

        self.data_frame = pd.DataFrame()

        self.model = None
        pass

    def setup_tf(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
            # self.model = SiameseLSTM(Config.FLAGS, vocab_size=len(self.embedding_matrix), batch_size=1024,
            #                          enable_summary=True)
            # self.model.get_predictions()

    def load_variables(self):
        with self.graph.as_default():
            with self.sess.as_default():
                # Load the saved meta graph and restore variables
                print("Loading TF checkpoint from: {}".format(self.checkpoint_file))
                print("Loading TF checkpoint from: {}.meta".format(self.checkpoint_file))

                self.saver = tf.train.import_meta_graph("{}.meta".format(self.checkpoint_file))


                self.sess.run(tf.global_variables_initializer())
                self.saver.restore(self.sess, self.checkpoint_file)

                # Get the placeholders from the graph by name
                self.input_x1 = self.graph.get_operation_by_name('input-layer/input_x1').outputs[0]
                self.input_x2 = self.graph.get_operation_by_name('input-layer/input_x2').outputs[0]
                self.embedding_placeholder = self.graph.get_operation_by_name('embed-layer/embedding_placeholder').outputs[0]
                self.dropout_keep_prob = self.graph.get_operation_by_name('dropout_keep_prob').outputs[0]
                self.is_training = self.graph.get_operation_by_name('input-layer/is_training').outputs[0]

                # Tensors we want to evaluate
                # self.logits = self.graph.get_operation_by_name('siamese_network/pred_layer/logits').outputs[0]
                self.predictions = self.graph.get_operation_by_name('prediction-layer/predictions').outputs[0]

    def predict(self):

            self.setup_tf()
            self.load_variables()

            # Collect the predictions here
            all_predictions = []
            all_ids = []
            i = 0
            for batch in tqdm(self.test_batches):
                i += 1
                x1, x2, id = zip(*batch)
                feed_dict = {
                    self.input_x1: x1,
                    self.input_x2: x2,
                    self.embedding_placeholder: self.embedding_matrix,
                    self.dropout_keep_prob: 1.0,
                    self.is_training: False
                }

                batch_predictions = self.sess.run([self.predictions], feed_dict)
                all_ids.extend(id)
                if(len(batch_predictions) == 1): #retured value is array of logits, consider second column
                    all_predictions.extend((batch_predictions[0][:,-1].tolist()))
                    # all_predictions.extend((batch_predictions[0].tolist()))
                else:
                    all_predictions.extend(batch_predictions)
                # if(i > 5): break
                # return all_ids, all_predictions

            self.data_frame['test_id'] = np.asarray(all_ids)
            self.data_frame['is_duplicate'] = np.asarray(all_predictions)
            self.data_frame = self.data_frame.sort(['test_id'])
            self.data_frame.to_csv(self.csv_file_name, index=False)
            return self.data_frame