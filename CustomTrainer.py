import ray
import os
from ray import tune
from ray.rllib.models import ModelCatalog

from hunter_dqn.dqn_model import DQNModelHunter
from hunter_dqn.hunters import DQNTrainer

if __name__ == "__main__":
    ray.init()
    ModelCatalog.register_custom_model("DQNModelHunter", DQNModelHunter)

    tune.run(
        DQNTrainer,
        # checkpoint_freq=10,
        checkpoint_at_end=True,
        stop={"timesteps_total": 2000},
        config={
            "num_gpus": 0,
            "num_workers": 1,
            "framework": "torch",
            # "sample_batch_size": 50,
            "env": "MultiHunterEnv-v0",

            ########################################
            # Parameters Agent
            ########################################
            "lr": 4e-3,
            # "lr": tune.grid_search([5e-3, 2e-3, 1e-3, 5e-4]),
            "gamma": 0.985,
            # "gamma": tune.grid_search([0.983, 0.985, 0.986, 0.987, 0.988, 0.989]),
            "epsilon": 1,
            "epsilon_decay": 0.99998,
            "epsilon_min": 0.01,
            "buffer_size": 20000,
            "batch_size": 2000,

            "dqn_model": {
                "custom_model": "DQNModelHunter",
                "custom_model_config": {
                    # "network_size": [32, 64, 32],
                },  # extra options to pass to your model
            },


            ########################################
            # Evaluation parameters
            ########################################
            "evaluation_interval": 100, # based on training iterations
            "evaluation_num_episodes": 100,
            "evaluation_config": {
                "epsilon": 1,
            },
        }
    )