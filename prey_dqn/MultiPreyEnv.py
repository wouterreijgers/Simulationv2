from typing import Tuple, Dict, Union, Any

from gym import register
from ray.rllib.env import MultiAgentEnv
from ray.rllib.train import torch
from ray.rllib.utils.typing import MultiAgentDict


import numpy as np

from prey_dqn.prey_env import PreyEnv


class MultiPreyEnv(MultiAgentEnv):
    def __init__(self, config):
        num = config.pop("num_agents", 1)
        self.agents = [PreyEnv() for _ in range(num)]
        self.dones = []
        self.observation_space = self.agents[0].observation_space
        self.action_space = self.agents[0].action_space
        self.action_shape = self.agents[0].action_space.n
        self.alive = 0

    def reset(self) -> MultiAgentDict:
        self.dones = []
        return {"prey_"+str(i): a.reset() for i, a in enumerate(self.agents)}

    def step(self, action_dict: MultiAgentDict) -> Tuple[dict, dict, Dict[Union[str, Any], Union[bool, Any]], dict]:
        preys = []
        n = 0
        print(len(self.agents))
        observation, reward, done, reproduce = {}, {}, {}, {}
        alive = []
        print(len(action_dict), action_dict)
        for i, action in action_dict.items():
            if not i in self.dones:
                observation[i], reward[i], done[i], reproduce[i] = self.agents[i].step(action)
                if done[i]:
                    self.dones.append(i)
                alive.append(i)

        for i in alive:
            # print("len", observation, action_dict[0], reward)
            if not i in self.dones:
                if reproduce[i]:
                    new_prey = PreyEnv()
                    observation[len(self.agents)] = new_prey.reset()
                    reward[len(self.agents)] = 0
                    done[len(self.agents)] = False
                    reproduce[len(self.agents)] = False
                    self.agents.append(new_prey)
        done["__all__"] = len(self.dones) == len(self.agents)
        print(observation)
        self.alive = len(observation)
        return observation, reward, done, reproduce

    def get_positions(self, obs):
        obs_batch = torch.tensor([])
        for i, obs in obs.items():
            if not i in self.dones:
                #print('get pos loop ', i)
                obs_batch = torch.cat([obs_batch, torch.tensor(self.agents[i].get_position()).unsqueeze(0)], 0)
        #print("obs_batch", len(obs_batch), obs_batch)
        return obs_batch

    def get(self):
        return self

#    @property
#    def unwrapped(self):
#        """Completely unwrap this env.
#        Returns:
#            gym.Env: The base non-wrapped gym.Env instance
#        """
#        return self
