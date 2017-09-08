## Gym
import gym
import numpy as np
import logging
logger = logging.getLogger(__name__)

from dhira.data.features.cartpole_feature import CartPoleFeature


class CartPole():
    def __init__(self):
        self._env = gym.make('CartPole-v0')
        self.reset()
        self.running_reward = None
        self.reward_sum = 0
        self.episode_number = 1
        self.total_episodes = 10000
        self._is_done = False
        self._reward = 0
        self._action = 0

    def render(self):
        self._env.render()

    def reset(self):
        self._is_done = False
        return self._env.reset()

    def close(self):
        self._env.close()

    @property
    def is_done(self):
        return self._is_done

    @property
    def reward(self):
        return self._reward

    @property
    def action(self):
        return self._action

    def reset_get_initial_observation(self):
        obs = self.reset()
        # The observation is the previous state of the environment,
        # which was used to make the action and determine the reward for the action.
        return CartPoleFeature(action=None, observation=obs, reward=None, done=None)

    def get_next_action(self, model_prediction):
        # print(model_prediction)
        return 1 if np.random.uniform() < model_prediction else 0
        # return 1 if model_prediction > 0.5 else 0


    def next_action(self, model_prediction):
        action = self.get_next_action(model_prediction)


        observation, reward, done, info = self._env.step(action)
        self.observation = observation
        self._is_done = done
        self._reward = reward
        self._action = action
        # y = 1 if action == 0 else 0
        #action at t, observation at t-1, reward at t
        return CartPoleFeature(action=action, observation=self.observation, reward=self._reward, done=self._is_done)



