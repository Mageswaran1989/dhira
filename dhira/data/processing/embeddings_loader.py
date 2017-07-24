import pickle

import mmap
from tqdm import tqdm
import numpy as np
import spacy
import en_core_web_md

from dhira.global_config import Config, NLPEngine

def get_line_number(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

class EmbeddingLoader:
    def get(self, nlp_engine=NLPEngine.SPACY):
        if not isinstance(nlp_engine, NLPEninge):
            raise TypeError("config should of type dhira.global_config.NLPEninge")

        if(nlp_engine == NLPEngine.NONE):
            return DefaultEmbeddingLoader
        elif(nlp_engine == NLPEngine.SPACY):
            pass
        else:
            raise EnvironmentError("Select a NLPEngine using dhira.global_config.NLPEninge")

class EmbeddingFactory(object):
    def __init__(self, word_vec_file_path, embedding_save_path, word_index):
        object.__init__(self)
        self.word_vec_file_path = word_vec_file_path
        self.embedding_save_path = embedding_save_path
        self.word_index = word_index

    def load_glove_embeddings(self, vocab, num_unknown=100):
        # num_unknown: Specify the number of unknown words for putting in the embedding matrix
        if not isinstance(vocab, spacy.vocab.Vocab):
            raise TypeError("The input `vocab` must be type of 'spacy.vocab.Vocab', not %s." % type(vocab))

        max_vector_length = max(lex.rank for lex in vocab) + 1  # index start from 1
        matrix = np.zeros((max_vector_length + num_unknown + 2, vocab.vectors_length),
                          dtype='float32')  # 2 for <PAD> and <EOS>

        # Normalization
        for lex in vocab:
            if lex.has_vector:
                matrix[lex.rank + 1] = lex.vector / lex.vector_norm

        return matrix

    def pickle_embeddings(self, pickle_path=None):
        nlp = en_core_web_md.load()
        embedding_matrix = self.load_glove_embeddings(nlp.vocab)
        pass

    def load_embeddings(self, embeddings_save_path=None):
        pass

class SpacyEmdeddingLoader(EmbeddingFactory):
    def __init__(self):
        pass

    def pickle_embeddings(self, pickle_path=None):
        pass

    def load_embeddings(self, embeddings_save_path=None):
        pass

class DefaultEmbeddingLoader(EmbeddingFactory):
    def __init__(self, word_vec_file_path=Config.EMBEDDING_PATH,
                 embeddings_save_path='embedding_matrix.p',
                 word_index = None):
        if word_index is None:
            word_index =  pickle.load(open('word_index.p', 'rb'))
        else:
            word_index = word_index

        EmbeddingFactory.__init__(self, word_vec_file_path, embeddings_save_path, word_index)

    def pickle_embeddings(self, pickle_path="embedding_matrix.p"):
        print('Reading word2vec/word2vec vectors from : ', self.word2vec_path)
        f = open(self.word2vec_path)

        embeddings_index = {}
        for line in tqdm(f, total=get_line_number(self.word2vec_path)):
            values = line.split()
            word = ''.join(values[:-300])
            coefs = np.asarray(values[-300:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        print('Found %s word vectors.' % len(embeddings_index))

        embedding_matrix = np.zeros((len(self.word_index) + 1, 300))
        null_word_embeddings = 0
        for word, i in tqdm(self.word_index.items()):
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                print('Missed word: ', word)
                null_word_embeddings += 1

        print("Null word embeddings: ", null_word_embeddings)
        pickle.dump(embedding_matrix, open(pickle_path, "wb"))
        return embedding_matrix

    @staticmethod
    def load_embeddings(embeddings_save_path='embedding_matrix.p'):
        embedding_matrix = pickle.load(open(embeddings_save_path, "rb"))
        return embedding_matrix
