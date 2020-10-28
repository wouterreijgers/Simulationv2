import random
from typing import Union, List, Optional, Dict, Tuple

import torch
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import ModelWeights, TensorType


class PreyPolicy(Policy):
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self.action_space = action_space
        self.observation_space = observation_space
        self.config = config
        self.action_shape = action_space.n
        self.dtype_f = torch.FloatTensor

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        explore=None,
                        timestep=None,
                        **kwargs):
        # TODO: for now the hunter does random moves
        obs_batch_t = torch.tensor(obs_batch).type(self.dtype_f)
        action_batch_t = random.randint(0, self.action_shape - 1)
        return [action_batch_t], [], {}

    def learn(self, samples):
        pass
