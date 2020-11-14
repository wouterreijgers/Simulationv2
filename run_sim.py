import ray
import json
import gym
import numpy as np

from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune import register_env

from MultiAgentSimEnv import MultiAgentSimEnv
from hunter_dqn.dqn_model import DQNModelHunter
from hunter_dqn.hunters import DQNTrainer


def env_creator(env_config):
    return MultiAgentSimEnv(env_config)


def policy_mapping_fn(agent_id):
    if agent_id.startswith("hunter"):
        return "hunter"
    else:
        return "prey"


if __name__ == "__main__":

    # Settings
    # folder = "/home/wouter/ray_results/DQNAlgorithm/DQNAlgorithm_CartPole-v1_0cf22_00000_0_2020-10-26_17-05-21"
    env_name = "MultiAgentSim-v0"
    checkpoint = 1000
    num_episodes = 1
    ray.init()
    config = {
        "batch_size": 2000,
        "buffer_size": 20000,
        "dqn_model": {
            "custom_model": "DQNModelHunter",
            "custom_model_config": {}
        },
        "env": "MultiAgentSimEnv-v0",
        "epsilon": 1,
        "epsilon_decay": 0.99998,
        "epsilon_min": 0.01,
        "evaluation_config": {
            "epsilon": 1
        },
        "evaluation_interval": 100,
        "evaluation_num_episodes": 100,
        "framework": "torch",
        "gamma": 0.985,
        "lr": 0.004,
        "num_gpus": 0,
        "num_workers": 1
    }
    env = register_env("MultiAgentSimEnv-v0", env_creator)
    ModelCatalog.register_custom_model("DQNModel", DQNModelHunter)

    trainer = DQNTrainer(env=env,
                         config=config)

    avg_reward = 0
    for episode in range(num_episodes):
        step = 0
        total_reward = 0
        done = False
        observation = env.reset()

        while not done:
            step += 1
            # env.render()
            print(observation)
            action, _, _ = trainer.get_policy().compute_actions([observation], [])
            observation, reward, done, info = env.step(action[0])
            total_reward += reward
        print("episode {} received reward {} after {} steps".format(episode, total_reward, step))
        avg_reward += total_reward
    print('avg reward after {} episodes {}'.format(avg_reward / num_episodes, num_episodes))
    env.close()
    del trainer
