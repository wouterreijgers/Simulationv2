from typing import Tuple

import gym
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

from hunter_dqn.MultiHunterEnv import MultiHunterEnv
from hunter_dqn.hunter_env import HunterEnv
from prey_dqn.MultiPreyEnv import MultiPreyEnv
from prey_dqn.prey_env import PreyEnv


class MultiAgentSimEnv(MultiAgentEnv):
    def __init__(self, config):
        # Hunters
        num = config.pop("num_hunters", 20)
        self.agents = [HunterEnv() for _ in range(num)]
        self.dones = []
        self.observation_space_hunter = HunterEnv().observation_space
        self.observation_space = self.observation_space_hunter
        self.action_space_hunter = HunterEnv().action_space
        self.action_space = self.action_space_hunter

        self.action_shape = HunterEnv().action_space.n
        self.alive = 0

        num = config.pop("num_preys", 100)
        self.prey_agents = [PreyEnv() for _ in range(num)]
        self.dones = []
        self.observation_space_prey = PreyEnv().observation_space
        self.action_space_prey = PreyEnv().action_space
        self.action_shape = HunterEnv().action_space.n
        self.alive = 0

    def reset(self) -> MultiAgentDict:
        self.dones = []
        obs_batch = {}
        for i, a in enumerate(self.agents):
            obs_batch["hunter_"+str(i)] = a.reset()
        for i, a in enumerate(self.prey_agents):
            obs_batch["prey_"+str(i)] = a.reset()
        print(obs_batch)
        return obs_batch

    def step(self, action_dict: MultiAgentDict) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
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
                    if "hunter" in i:
                        new_agent = HunterEnv()
                    else:
                        new_agent = PreyEnv()
                    observation[len(self.agents)] = new_agent.reset()
                    reward[len(self.agents)] = 0
                    done[len(self.agents)] = False
                    reproduce[len(self.agents)] = False
                    self.agents.append(new_agent)
        done["__all__"] = len(self.dones) == len(self.agents)
        print(observation)
        self.alive = len(observation)
        return observation, reward, done, reproduce
