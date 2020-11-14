import os
import random

import ray
from ray import tune
from ray.rllib.examples.models.shared_weights_model import TorchSharedWeightsModel, SharedWeightsModel1, \
    SharedWeightsModel2
from ray.rllib.models import ModelCatalog

from hunter_dqn.MultiHunterEnv import MultiHunterEnv
from hunter_dqn.hunter_env import HunterEnv
from hunter_dqn.dqn_model import DQNModel_Hunter
from hunter_dqn.hunters import DQNTrainer

from simulation.environment import Environment

import argparse
import os

import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--torch", action="store_true")


def Hunter_trainer(config, reporter):
    multi_hunter_trainer = PPOTrainer(MultiHunterEnv, config)
    for _ in range(100):
        environment.simulate()
        result = multi_hunter_trainer.train()
        result["phase"] = 1
        reporter(**result)
        phase1_time = result["timesteps_total"]
    state = multi_hunter_trainer.save()
    multi_hunter_trainer.stop()


if __name__ == '__main__':
    training = True
    ray.init()

    ModelCatalog.register_custom_model("DQNModel", DQNModel_Hunter)
    config_hunter = {
        "num_gpus": 0,
        "num_workers": 1,
        "framework": "torch",
        "lr": 4e-3,
        # "lr": tune.grid_search([5e-3, 2e-3, 1e-3, 5e-4]),
        "gamma": 0.985,
        # "gamma": tune.grid_search([0.983, 0.985, 0.986, 0.987, 0.988, 0.989]),
        "epsilon": 1,
        "epsilon_decay": 0.99998,
        "epsilon_min": 0.01,
        "buffer_size": 20000,
        "batch_size": 2000,

        "env": MultiHunterEnv,
        "env_config": {
            "num_agents": 20,
            "energy_to_reproduce": 30,
            "energy_per_prey_eaten": 30,
            "max_age": 30,
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "model": {
            "custom_model": "DQNModel",
        },
        "vf_share_layers": True,
    }
    if training:
        resources = PPOTrainer.default_resource_request(config_hunter).to_json()
        tune.run(Hunter_trainer, resources_per_trial=resources)

    # Define environment
    environment = Environment()
    #environment.create_hunters()
    environment.create_preys()
    environment.simulate()
