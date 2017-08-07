import codecs
import random
import numpy as np
import io
from tqdm import tqdm

from dhira.data.dataset.text import TextDataset
from dhira.data.features.movie_review_feature import MovieReviewFeature

class MovieReview(TextDataset):

    def __init__(self,
                 nlp,
                 name='movie-review-data',
                 feature_type = MovieReviewFeature,
                 max_lengths={"num_sentence_words": 50},
                 pad=True,
                 download_path=None):
        """
        
        :param nlp: spaCy pipeline. Eg: nlp = spacy.load('en_core_web_md')     
        :param name: str default is movie-review-data
        :param feature_type: deafule is MovieReviewFeature
        :param download_path: 
        """

        super(MovieReview, self).__init__(name=name,
                                           feature_type=feature_type,
                                           download_path=download_path)

        random.seed(42)
        self.nlp = nlp
        self.max_lengths = max_lengths
        self.pad = pad

        # We now need to check if the user specified max_length for
        # the feature, and accordingly truncate or pad if applicable. If
        # max_length is None for a given string key, we assume that no
        # truncation is to be done and the max lengths should be read from the
        # features.
        if not self.pad and self.max_lengths:
            raise ValueError("Passed in max_length {}, but set pad to false. "
                             "Did you mean to do this?".format(self.max_lengths))

        # sentence polarity dataset v1.0 (includes sentence polarity dataset README v1.0:
        #   5331 positive and 5331 negative processed sentences / snippets.
        # Introduced in Pang/Lee ACL 2005. Released July 2005.
        self._url = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'

        #List of all lines read form the dataset appended with lables
        self.features = []

        self.downloaded_path = self.download(self._url)
        self.preprocess_data()

    def preprocess_data(self):

        postive_reviews_file = self.downloaded_path + 'rt-polaritydata/rt-polarity.pos'
        negative_reviews_file = self.downloaded_path + 'rt-polaritydata/rt-polarity.neg'

        # for line in tqdm(codecs.open(postive_reviews_file, "r").readlines()):
        #     line = line.decode('ISO-8859-1').encode('ascii',errors='ignore').decode()
        for line in tqdm(io.open(postive_reviews_file, encoding="ISO-8859-1")):
            self.features.append('0\\' + line) #TODO find a better way

        # for line in tqdm(codecs.open(negative_reviews_file, "r").readlines()):
        #     line = line.decode('ISO-8859-1').encode('ascii', errors='ignore').decode()
        for line in tqdm(io.open(negative_reviews_file, encoding="ISO-8859-1")):
            self.features.append('1\\' + line)

        self.features = [self.feature_type.read_from_line(line, self.nlp) for line in tqdm(self.features)]

        #Split the data as 8:1:1 for train:test:val
        self.split_dataset(self.features)

    def get_train_batch_generator(self):
        for feature in self.train_features:
            # For each instance, we want to pad or truncate if applicable
            if self.pad:
                feature.pad(self.max_lengths)
            # Now, we want to take the instance and convert it into
            # NumPy arrays suitable for training.
            inputs, labels = feature.as_training_data()
            yield inputs, labels

    def get_validation_batch_generator(self):
        for feature in self.val_features:
            # For each instance, we want to pad or truncate if applicable
            if self.pad:
                feature.pad(self.max_lengths)
            # Now, we want to take the instance and convert it into
            # NumPy arrays suitable for training.
            inputs, labels = feature.as_training_data()
            yield inputs, labels

    def get_test_batch_generator(self):
        for feature in self.test_features:
            # For each instance, we want to pad or truncate if applicable
            if self.pad:
                feature.pad(self.max_lengths)
            # Now, we want to take the instance and convert it into
            # NumPy arrays suitable for training.
            inputs, labels = feature.as_training_data()
            yield inputs, labels
