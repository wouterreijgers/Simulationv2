import math
import random
import queue

import ray
from ray.rllib.train import torch

from prey_dqn.prey_env import PreyEnv
from prey_dqn.prey_policy import PreyPolicy
from util.config_reader import ConfigReader


class Preys():
    def __init__(self):
        self.steps = 0
        self.preys = []
        self.hunter_batch = torch.tensor([])

        self.folder = "/home/wouter/ray_results/DQNAlgorithm/DQNAlgorithm_CartPole-v1_0cf22_00000_0_2020-10-26_17-05-21"
        self.checkpoint = 1000
        self.env_name = "PreyEnv"
        self.total_reward = 0
        self.hunters = None
        self.trainer = None
        # ModelCatalog.register_custom_model("DQNModel", DQNModel)
        # with open(self.folder + "/params.json") as json_file:
        #    config = json.load(json_file)
        # self.trainer = DQNTrainer(env=self.env_name, config=config)

    def set_hunters(self, hunters):
        self.hunters = hunters
        self.trainer = PreyPolicy(PreyEnv(hunters).observation_space, PreyEnv(hunters).action_space, {})

    def update_preys(self):
        living_preys = []
        for i in range(len(self.preys)):
            env, observation = self.preys.pop()
            self.steps += 1
            # action, _, _ = self.trainer.get_policy().compute_actions([observation], [])
            action, _, _ = self.trainer.compute_actions([observation], [])
            observation, reward, done, reproduce = env.step(action[0])
            #print(observation, reward, done, reproduce)
            # print(reproduce)
            self.total_reward += reward
            if not done:
                living_preys.append([env, observation])
            if reproduce:
                living_preys.append(self.new_prey())
        self.preys = living_preys

    def has_preys(self):
        if len(self.preys) > 0:
            return True
        return False

    def new_prey(self):
        # env = gym.make(self.env_name)
        env = PreyEnv(self.hunters)
        observation = env.reset()
        self.preys.append([env, observation])
        return [env, observation]

    def get_preys(self):
        obs_batch = torch.tensor([])
        for env, observation in self.preys:
            obs_batch = torch.cat([obs_batch, torch.tensor(env.get_position()).unsqueeze(0)], 0)
        return obs_batch

    def get_rel_x_y(self, state):
        best_dist = math.inf
        rel_x = 0
        rel_y = 0
        if len(self.preys) == 0:
            return math.inf, math.inf
        for env, obs in self.preys:
            pos = env.get_position()
            dist = abs(state[0] - pos[0].item()) + abs(state[1] - pos[1].item())
            if dist < best_dist:
                best_dist = dist
                rel_x = pos[0].item() - state[0]
                rel_y = pos[1].item() - state[1]
        return rel_x, rel_y
