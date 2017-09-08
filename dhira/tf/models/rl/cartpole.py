#https://github.com/ashutoshkrjha/Cartpole-OpenAI-Tensorflow


import numpy as np
import _pickle as pickle
import tensorflow as tf

import matplotlib.pyplot as plt
import math
from overrides import overrides


#TensorFlow
from dhira.tf.models.internal.base_tf_model import BaseTFModel

class PolicyGradient(BaseTFModel):
    def __init__(self,
                 name='PlocyGradient',
                 run_id=0,
                 save_dir=None,
                 log_dir=None):
        super(self.__class__, self).__init__(name=name,
                 run_id=run_id,
                 save_dir=save_dir,
                 log_dir=log_dir)

        # Hyperparameters
        self.H_SIZE = 10  # Number of hidden layer neurons
        self.batch_size = 5  # Update Params after every 5 episodes
        self.ETA = 1e-2  # Learning Rate
        self.GAMMA = 0.99  # Discount factor

        self.INPUT_DIM = 4  # Input dimensions

    def _create_placeholders(self):
        # Network to define moving left or right
        self.observations = tf.placeholder(tf.float32, [None, self.INPUT_DIM], name="observations")
        self.actions = tf.placeholder(tf.float32, [None, 1], name="action")
        # self.reward = tf.placeholder(tf.float32, name="reward_signal")
        self.rewards = tf.placeholder(tf.float32, [None, 1], name="reward_signal")


    @overrides
    def _setup_graph_def(self):
        layer = tf.layers.dense(inputs=self.observations,
                                units=8,
                                activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.pred_action = tf.layers.dense(inputs=layer,
                                           units=1,
                                           activation=tf.nn.sigmoid,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer())

        # The loss function. This sends the weights in the direction of making actions
        # that gave good advantage (reward over time) more likely, and actions that didn't less likely.
        # log(y * (y - y^) + (1 - y) * (y + y^))
        # self._loglik = tf.log(self.action * (self.action - self.pred_action) +
        #                 (1 - self.action) * (self.action + self.pred_action))
        # self._loss = -tf.reduce_mean(self._loglik * self.reward)

        #  mean(log(y * log(y^) + (1 - y) * log(1 - y^))) * rewards
        self._loss = - tf.reduce_mean((self.actions * tf.log(self.pred_action) +
                                      (1 - self.actions) * (tf.log(1 - self.pred_action))) * self.rewards,0)


        self._optimizer = tf.train.AdamOptimizer(learning_rate=self.ETA).minimize(self._loss, global_step=self.global_step)  # Adam optimizer

    @overrides
    def _get_eval_metric(self):
        return self._loss

    @overrides
    def _get_prediction(self):
        return self.pred_action

    @overrides
    def _get_optimizer(self):
        return self._optimizer

    @overrides
    def _get_loss(self):
        return self._loss

    # def discount_rewards(self, r, GAMMA=0.99):
    #     """ take 1D float array of rewards and compute discounted reward """
    #     discounted_r = np.zeros_like(r)
    #     running_add = 0
    #     for t in reversed(range(0, r.size)):
    #         running_add = running_add * GAMMA + r[t]
    #         discounted_r[t] = running_add
    #     return discounted_r

    def discount_rewards(self, rewards, gamma):
        """
        Return discounted rewards weighed by gamma.
        Each reward will be replaced with a weight reward that
        involves itself and all the other rewards occuring after it.
        The later the reward after it happens, the less effect it
        has on the current rewards's discounted reward since gamma&amp;lt;1.

        [r0, r1, r2, ..., r_N] will look someting like:
        [(r0 + r1*gamma^1 + ... r_N*gamma^N), (r1 + r2*gamma^1 + ...), ...]
        """
        return np.array([sum([gamma ** t * r for t, r in enumerate(rewards[i:])])
                         for i in range(len(rewards))])

    @overrides
    def _get_train_feed_dict(self, batch, is_done):
        inputs, lables = batch
        observation, reward = inputs
        if is_done is True:
            # Compute the discounted reward

            reward = np.vstack(
                self.discount_rewards(reward, self.GAMMA))

            discounted_epr = self.discount_rewards(reward, self.GAMMA)

            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            return {self.actions:lables[0], self.observations: observation[0], self.rewards:discounted_epr}

        else:
            return {self.observations: observation[0]}