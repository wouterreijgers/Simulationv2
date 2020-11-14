import os
import random
from collections import deque
from typing import Union, List, Optional, Dict, Tuple
import numpy as np

import torch
from ray.rllib.models import ModelCatalog
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import ModelWeights, TensorType
from torch.nn import MSELoss


class HunterPolicy(Policy):
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config
        self.action_shape = action_space.n

        # GPU settings
        os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.dtype_f = torch.FloatTensor
        self.dtype_l = torch.LongTensor
        self.dtype_b = torch.BoolTensor
        self.lr = self.config["lr"]  # Extra options need to be added in dqn.py
        self.epsilon = self.config["epsilon"]
        self.epsilon_decay = self.config["epsilon_decay"]
        self.epsilon_min = self.config["epsilon_min"]
        self.gamma = torch.tensor(self.config["gamma"]).to(self.device, non_blocking=True)
        self.batch_size = self.config["batch_size"]

        self.memory = deque(maxlen=self.config["buffer_size"])

        self.dqn_model = ModelCatalog.get_model_v2(
            obs_space=self.observation_space,
            action_space=self.action_space,
            num_outputs=2,
            name="DQNModel",
            model_config=self.config["dqn_model"],
            framework="torch",
        ).to(self.device, non_blocking=True)
        self.MSE_loss_fn = MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.dqn_model.parameters(), lr=self.lr)

    def experience_buffer(self, samples):
        obs = samples["obs"]
        new_obs = samples["new_obs"]
        rewards = samples["rewards"]
        actions = samples["actions"]
        dones = samples["dones"]

        batch = zip(obs, new_obs, rewards, actions, dones)
        for obs_s, new_obs_s, rewards_s, actions_s, dones_s in batch:
            self.memory.append([obs_s, new_obs_s, rewards_s, actions_s, dones_s])

    def sample_from_memory(self):
        batch = random.sample(self.memory, self.batch_size)
        samples = {}
        samples["obs"] = [None for _ in range(self.batch_size)]
        samples["new_obs"] = [None for _ in range(self.batch_size)]
        samples["rewards"] = [None for _ in range(self.batch_size)]
        samples["actions"] = [None for _ in range(self.batch_size)]
        samples["dones"] = [None for _ in range(self.batch_size)]

        i = 0
        for sample in batch:
            samples["obs"][i] = sample[0]
            samples["new_obs"][i] = sample[1]
            samples["rewards"][i] = sample[2]
            samples["actions"][i] = sample[3]
            samples["dones"][i] = sample[4]
            i += 1
        return samples

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
        # Worker function
        print(obs_batch)
        obs_batch_t = torch.tensor(obs_batch).type(self.dtype_f)
        q_value_batch_t = self.dqn_model(obs_batch_t)
        action_batch_t = torch.argmax(q_value_batch_t, axis=1)

        epsilon_log = []
        for index in range(len(action_batch_t)):
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min
            epsilon_log.append(self.epsilon)
            if np.random.random() < self.epsilon:
                action_batch_t[index] = random.randint(0, self.action_shape - 1)

        action = action_batch_t.cpu().detach().tolist()
        return action, [], {"epsilon_log": epsilon_log}

    def learn_on_batch(self, samples):
        #print('learn')
        pass

    def get_weights(self):
        # Trainer function
        #print('get_weights test')
        weights = {}
        weights["dqn_model"] = self.dqn_model.cpu().state_dict()
        self.dqn_model.to(self.device, non_blocking=False)
        return weights

    def set_weights(self, weights):
        #print('set_weights')
        # Worker function
        if "dqn_model" in weights:
            self.dqn_model.load_state_dict(weights["dqn_model"], strict=True)
            self.dqn_model.to(self.device, non_blocking=False)
