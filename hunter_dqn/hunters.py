import json
import math

import gym
import ray
import numpy as np
from ray.rllib.agents import with_common_config
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.models import ModelCatalog
from ray.rllib.train import torch

from hunter_dqn.MultiHunterEnv import MultiHunterEnv
from hunter_dqn.hunter_env import HunterEnv
#from hunter_dqn.hunter_model import DQNModelHunter
from hunter_dqn.hunter_policy import HunterPolicy


class Hunters():
    def __init__(self):
        self.steps = 0
        self.hunters = []
        self.env = MultiHunterEnv({"num_agents": 20})
        self.hunter_batch = torch.tensor([])

        self.folder = "/home/wouter/ray_results/DQNAlgorithm/DQNAlgorithm_CartPole-v1_0cf22_00000_0_2020-10-26_17-05-21"
        self.checkpoint = 1000
        self.env_name = "HunterEnv"
        self.total_reward = 0
        #self.preys = preys
        self.dtype_f = torch.FloatTensor

        policy_config = {
            "lr": 4e-3,
            # "lr": tune.grid_search([5e-3, 2e-3, 1e-3, 5e-4]),
            "gamma": 0.985,
            # "gamma": tune.grid_search([0.983, 0.985, 0.986, 0.987, 0.988, 0.989]),
            "epsilon": 1,
            "epsilon_decay": 0.99998,
            "epsilon_min": 0.01,
            "buffer_size": 20000,
            "batch_size": 2000,
        }
        #ModelCatalog.register_custom_model("DQNModel", DQNModel)
        #with open(self.folder + "/params.json") as json_file:
        #    config = json.load(json_file)
        #self.trainer = DQNTrainer(env=self.env_name, config=config)
        #self.trainer = HunterPolicy(HunterEnv(preys).observation_space, HunterEnv(preys).action_space, policy_config)
        self.obs = self.env.reset()

    def update_hunters(self):
        # action, _, _ = self.trainer.compute_actions(self.obs, [])
        # observation, reward, done, reproduce = self.env.step(action)
        # print([observation, reward, done, reproduce])
        # self.obs = observation

        living_hunters = []
        # for i in range(len(self.hunters)):
        #     env, observation = self.hunters.pop()
        #     self.steps += 1
        #     # action, _, _ = self.trainer.get_policy().compute_actions([observation], [])
        #     action, _, _ = self.trainer.compute_actions([observation], [])
        #     observation, reward, done, reproduce = env.step(action[0])
        #     #print(reproduce)
        #     self.total_reward += reward
        #     if not done:
        #         living_hunters.append([env, observation])
        #     if reproduce:
        #         print("birth", self.steps)
        #         living_hunters.append(self.new_hunter())
        # self.hunters = living_hunters

    def has_hunters(self):
        if len(self.obs) > 0:
            return True
        return False

    def get_hunters(self):
        return self.env.get_positions(self.obs)

    def hunt(self, preys):
        for hunter in self.hunters:
            for prey in preys:
                if hunter.hunt(prey):
                    prey.kill()

    def get_rel_x_y(self, state):
        best_dist = math.inf
        rel_x = 0
        rel_y = 0
        if len(self.hunters)==0:
            return math.inf, math.inf
        for env, obs in self.hunters:
            pos = env.get_position()
            dist = abs(state[0] - pos[0].item()) + abs(state[1] - pos[1].item())
            if dist < best_dist:
                best_dist = dist
                rel_x = pos[0].item() - state[0]
                rel_y = pos[1].item() - state[1]
        return rel_x, rel_y


DEFAULT_CONFIG = with_common_config({
    ########################################
    # Parameters Agent
    ########################################
    "lr": 0,
    "gamma": 0.9,
    "epsilon": 1,
    "epsilon_decay": 0.9995,
    "epsilon_min": 0.05,
    "buffer_size": 10000,
    "batch_size": 500,

    "dqn_model": {
        "custom_model": "?",
        "custom_model_config": {
            "network_size": [32, 64, 32],
        },  # extra options to pass to your model
    }
})


