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
        self.hunter_count = num
        self.agents={}
        for i in range(num):
            self.agents['hunter_'+str(i)] = HunterEnv()
        self.dones = []
        self.observation_space_hunter = HunterEnv().observation_space
        self.observation_space = self.observation_space_hunter
        self.action_space_hunter = HunterEnv().action_space
        self.action_space = self.action_space_hunter

        self.action_shape = HunterEnv().action_space.n
        self.alive = 0

        num = config.pop("num_preys", 100)
        self.prey_count = num

        for i in range(num):
            self.agents['prey_'+str(i)] = PreyEnv()
        self.dones = []
        self.observation_space_prey = PreyEnv().observation_space
        self.action_space_prey = PreyEnv().action_space
        self.action_shape = HunterEnv().action_space.n
        self.alive = 0

    def reset(self) -> MultiAgentDict:
        self.dones = []
        obs_batch = {}
        print(self.agents)
        for i, a in self.agents.items():
            #if a.observation_space == self.observation_space:
            obs_batch[i] = a.reset()
            # else:
            #     obs_batch["prey_" + str(i)] = a.reset()
        # for i, a in enumerate(self.prey_agents):
        #     obs_batch["prey_"+str(i)] = a.reset()
        # print(obs_batch)
        return obs_batch

    def step(self, action_dict: MultiAgentDict) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        #print(len(self.agents))
        observation, reward, done, reproduce = {}, {}, {}, {}
        alive = []
        #print(len(action_dict), action_dict)
        for id, action in action_dict.items():
            #i = int(id.split('_')[1])

            if not id in self.dones:
                observation[id], reward[id], done[id], reproduce[id] = self.agents[id].step(action)
                if done[id]:
                    self.dones.append(id)
                alive.append(id)
                # else:
                #     observation[id], reward[id], done[id], reproduce[id] = self.prey_agents[id].step(action)
                #     if done[id]:
                #         self.dones.append(id)
                #     alive.append(id)

        for id in alive:
            # print("len", observation, action_dict[0], reward)
            if not id in self.dones:
                if reproduce[id]:
                    if "hunter" in id:
                        self.hunter_count +=1
                        new_agent = HunterEnv()
                        new_id = "hunter_"+str(self.hunter_count)
                    else:
                        self.prey_count +=1
                        new_agent = PreyEnv()
                        new_id = "prey_"+str(self.prey_count)

                    observation[new_id] = new_agent.reset()
                    reward[new_id] = 0
                    done[new_id] = False
                    reproduce[new_id] = False
                    self.agents[new_id]=new_agent
        done["__all__"] = len(self.dones) == len(self.agents)
        #print(observation)
        self.alive = len(observation)
        return observation, reward, done, reproduce
