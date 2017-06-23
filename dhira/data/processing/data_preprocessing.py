
from dhira.input_preprocessor import InputProcessor
from dhira.global_config import Config


def prepare_data():
    inp_processor = InputProcessor(Config.TRAIN_DATA_PATH, Config.TEST_DATA_PATH)
    inp_processor.pickle_data_sets()
    inp_processor.pickle_embeddings(Config.EMBEDDING_PATH, "embedding_matrix.p")
