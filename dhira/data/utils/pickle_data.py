import pickle
import os
import errno


class PickleData:

    def __init__(self, pickle_directory):
        self.pickle_directory = pickle_directory

    def check_pickle_exists(self, pickle_file):
        """
        Check whether the pickle file exists before loading
        :param pickle_file Pickle file to be checked
        :return: 'True' if exists or otherwise
        """
        return os.path.exists(self.pickle_directory + '/' + pickle_file)

    def write_pickle(self, python_object, file_name):
        """
        Pickles the processed features for faster loading
        :param file_name: File Name to store the pickle objects
        :return: Path of the file created
        """

        if (not os.path.exists(self.pickle_directory)):
            try:
                os.makedirs(self.pickle_directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        file = self.pickle_directory + '/' + file_name
        if (os.path.exists(file)):
            input('Found existing files!!! Press Enter to Continue...')

        file_handler = open(file, 'wb')
        pickle.dump(python_object, file_handler)

    def read_pickle(self, file_name):
        """
        Reads the pickled file and returns the python object
        :param file_name: File Name to read the pickle objects
        :return: Python object
        """
        file = self.pickle_directory + '/' + file_name
        file_handler = open(file, 'rb')
        return pickle.load(file_handler)