from dhira.data.features.glove_feature import GloveFeature
from dhira.data.dataset.glove import GloveDataset
from unittest import TestCase
from overrides import overrides

class TestGloveDataset(TestCase):

    @overrides
    def setUp(self):
        super(TestGloveDataset, self).setUp()
        self.glove_dataset = GloveDataset(train_files='../data/offline/glove/training.txt',
                                val_files='../data/offline/glove/validation.txt')

    def test_window(self):
        res = self.glove_dataset.window(['word', 'manage', 'maths',
                                         'dhira', 'tensorflow', 'train', 'glove'],
                            -2,0)
        assert len(res) == 1
        assert res[0] == 'word'

        res = self.glove_dataset.window(['word', 'manage', 'maths',
                                         'dhira', 'tensorflow', 'train', 'glove'],
                            6,8)
        assert len(res) == 1
        assert res[0] == 'glove'

        res = self.glove_dataset.window(['word', 'manage', 'maths',
                                         'dhira', 'tensorflow', 'train', 'glove'],
                            2,4)
        assert len(res) == 3
        assert res == ['maths', 'dhira', 'tensorflow']



    def tearDown(self):
        ''