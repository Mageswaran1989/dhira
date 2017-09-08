import numpy as np
import logging
logger = logging.getLogger(__name__)

from dhira.data.features.internal.feature_base import FeatureBase

class CartPoleFeature(FeatureBase):
    def __init__(self, action, observation, reward, done):
        super(self.__class__, self).__init__(np.reshape(action, [1]))
        self.observation: np.ndarray = np.reshape(observation, [1, 4])
        self.action = np.reshape(action, [1, 1])
        self.done = done
        self.reward = np.reshape(reward, [1])

    def as_training_data(self):
        return ((self.observation, self.reward), (self.label,))

    def __str__(self):
        return str(self.observation)# + '' + self.action + self.reward + None