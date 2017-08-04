
# coding: utf-8

# In[ ]:


import sys
import os
import math
import spacy
sys.path.append("../")
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.debug("test")


# In[ ]:


from dhira.data.data_manager import DataManager
from dhira.data.embedding_manager import EmbeddingManager
from dhira.tf.models.siamese.siamese_bilstm import SiameseBiLSTM


# In[ ]:


# from dhira.data.dataset.text import IndexedDataset
# from dhira.data.features.pair_feature import PairFeature
# quora_dataset = IndexedDataset(name='quora',
#                                train_files='../data/quora/processed/train_cleaned_train_split.csv',
#                                val_files='../data/quora/processed/train_cleaned_val_split.csv',
#                                test_files='../data/quora/processed/test_final.csv',
#                                feature_type=PairFeature,
#                                pad=True,
#                                pickle_dir='../models/',
#                                max_lengths={"num_sentence_words": 30})


# In[ ]:


from dhira.data.dataset.quora import QuoraDataset
from dhira.data.features.quora_feature import QuoraFeature
quora_dataset = QuoraDataset(name='quora-spacy',
                               train_files='../data/quora/processed/train_cleaned_train_split.csv',
                               val_files='../data/quora/processed/train_cleaned_val_split.csv',
                               test_files='../data/quora/processed/test_final.csv',
                               feature_type=QuoraFeature,
                               pad=True,
                               pickle_dir='../models/',
                               max_lengths={"num_sentence_words": 30})


# In[ ]:


data_manager = DataManager(dataset=quora_dataset)


# In[ ]:


get_train_data_gen, train_data_size = data_manager.get_train_data_from_file()


# In[ ]:


get_val_data_gen, val_data_size = data_manager.get_validation_data_from_file()


# In[ ]:


# embedding_manager = EmbeddingManager(quora_dataset.data_indexer, pickle_dir='../models/')
# embedding_matrix = embedding_manager.get_embedding_matrix(300,'../data/quora/external/glove.6B.300d.txt')
nlp = spacy.load('en_core_web_md')
embedding_matrix = EmbeddingManager.get_spacy_embedding_matrix(nlp)


# In[ ]:


from pympler import asizeof
asizeof.asizeof(embedding_matrix) / 1024 / 1024


# In[ ]:


model = ''
model = SiameseBiLSTM(mode='train',
             save_dir='../models/',
             log_dir='../logs/',
             run_id='0',
             word_vocab_size=embedding_matrix.shape[0], 
             word_embedding_dim=300, 
             word_embedding_matrix=embedding_matrix, 
             fine_tune_embeddings=True, 
             share_encoder_weights=True,
             rnn_output_mode='last',
             output_keep_prob=0.5,
             rnn_hidden_size=256)



# In[ ]:


model.build_graph()


# In[ ]:


batch_size = 256
num_epochs = 1
num_train_steps_per_epoch = int(math.ceil(train_data_size / batch_size))
num_val_steps = int(math.ceil(val_data_size / batch_size))


# In[ ]:


num_train_steps_per_epoch, num_val_steps


# In[ ]:


log_period = 50
val_period = 150
save_period = 200


# In[ ]:


model.train(get_train_instance_generator=get_train_data_gen,
            get_val_instance_generator=get_val_data_gen,
            batch_size=batch_size,
            num_train_steps_per_epoch=num_train_steps_per_epoch,
            num_epochs=num_epochs,
            num_val_steps=num_val_steps,
            log_period=log_period,
            val_period=val_period,
            save_period=save_period,
            patience=0)


# In[ ]:


# get_test_data_gen, test_data_size = data_manager.get_test_data_from_file()

