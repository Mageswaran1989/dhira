import pickle
from nltk.tokenize import sent_tokenize, word_tokenize
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
import pandas as pd
from tqdm import tqdm

from dhira.global_config import Config


class Tokenizer(object):
    def __init__(self):
        self.word_counts = {}
        self.document_count = 0
        self.word_docs = {}
        self.num_words = Config.VOCAB_SIZE
        self.word_index = {}

    @staticmethod
    def word_count_to_tsv(tsv_file_path, pickle_file_path='word_counts.p'):
        wc = pickle.load(open(pickle_file_path, 'rb'))
        df = pd.DataFrame(wc, columns=['Word', 'Frequency'])
        df.to_csv(tsv_file_path, sep='\t', index=False)

    def text_to_seq(self, texts):
        words_list = []
        for text in tqdm(texts):
            seq = word_tokenize(text=text)
            seq = self.word_list_cleanup(seq)
            words_list.append(seq)
        return words_list

    def fit_on_seq(self, words_list):
        """Updates internal vocabulary based on a list of texts.

        Required before using `words_to_sequences`.

        # Arguments
            texts: can be a list of strings,
                or a generator of strings (for memory-efficiency)
        """
        self.document_count = 0
        for seq in tqdm(words_list):
            self.document_count += 1
            for w in seq:
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1
            for w in set(seq):
                if w in self.word_docs:
                    self.word_docs[w] += 1
                else:
                    self.word_docs[w] = 1

        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)

        pickle.dump(wcounts, open('word_counts.p', 'wb'))

        sorted_voc = [wc[0] for wc in wcounts]
        # note that index 0 is reserved, never assigned to an existing word
        self.word_index = dict(list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))

    def seq_to_vec(self, words_list):
        """Transforms each text in texts in a sequence of integers.

        Only top "num_words" most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.

        # Arguments
            words_list: A list of word sequence.

        # Returns
            A list of sequences.
        """
        res = []
        for vect in self.seq_to_vec_generator(words_list):
            res.append(vect)
        return res

    def seq_to_vec_generator(self, words_list):
        """Transforms each text in texts in a sequence of integers.

        Only top "num_words" most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.

        # Arguments
            words_list: A list of word sequence.

        # Yields
            Yields individual sequences.
        """
        num_words = self.num_words
        for seq in tqdm(words_list):
            vect = []
            for w in seq:
                i = self.word_index.get(w)
                if i is not None:
                    if num_words and i >= num_words:
                        continue
                    else:
                        vect.append(i)
            yield vect

    def word_list_cleanup(self, words, remove_stopwords=False, stem_words=False):
        # Clean the text, with the option to remove stopwords and to stem words.

        # # Convert words to lower case and split them
        # text = text.lower().split()

        text = []

        # Optionally, remove stop words
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            text = [w for w in words if not w in stops]

        text = " ".join(words)

        # Clean the text
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)

        text = re.sub(r"[^A-Za-z0-9]", " ", text)
        text = re.sub(r"what's", "", text)
        text = re.sub(r"What's", "", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"I'm", "I am", text)
        text = re.sub(r" m ", " am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"60k", " 60000 ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e-mail", "email", text)
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"quikly", "quickly", text)
        text = re.sub(r" usa ", " America ", text)
        text = re.sub(r" USA ", " America ", text)
        text = re.sub(r" u s ", " America ", text)
        text = re.sub(r" uk ", " England ", text)
        text = re.sub(r" UK ", " England ", text)
        text = re.sub(r"india", "India", text)
        text = re.sub(r"switzerland", "Switzerland", text)
        text = re.sub(r"china", "China", text)
        text = re.sub(r"chinese", "Chinese", text)
        text = re.sub(r"imrovement", "improvement", text)
        text = re.sub(r"intially", "initially", text)
        text = re.sub(r"quora", "Quora", text)
        text = re.sub(r" dms ", "direct messages ", text)
        text = re.sub(r"demonitization", "demonetization", text)
        text = re.sub(r"actived", "active", text)
        text = re.sub(r"kms", " kilometers ", text)
        text = re.sub(r"KMs", " kilometers ", text)
        text = re.sub(r" cs ", " computer science ", text)
        text = re.sub(r" upvotes ", " up votes ", text)
        text = re.sub(r" iPhone ", " phone ", text)
        text = re.sub(r"\0rs ", " rs ", text)
        text = re.sub(r"calender", "calendar", text)
        text = re.sub(r"ios", "operating system", text)
        text = re.sub(r"gps", "GPS", text)
        text = re.sub(r"gst", "GST", text)
        text = re.sub(r"programing", "programming", text)
        text = re.sub(r"bestfriend", "best friend", text)
        text = re.sub(r"dna", "DNA", text)
        text = re.sub(r"III", "3", text)
        text = re.sub(r"the US", "America", text)
        text = re.sub(r"Astrology", "astrology", text)
        text = re.sub(r"Method", "method", text)
        text = re.sub(r"Find", "find", text)
        text = re.sub(r"banglore", "Banglore", text)
        text = re.sub(r" J K ", " JK ", text)

        # Optionally, shorten words to their stems
        if stem_words:
            text = text.split()
            stemmer = SnowballStemmer('english')
            stemmed_words = [stemmer.stem(word) for word in text]
            text = " ".join(stemmed_words)

        # Return a list of words
        return (text.split(' '))