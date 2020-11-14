import ray
from ray.rllib.agents import Trainer
from ray.rllib.agents.ppo import ppo
from ray.tune import tune
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

from MultiAgentSimEnv import MultiAgentSimEnv
from hunter_dqn.dqn import DQNTrainer
from hunter_dqn.dqn_model import DQNModel
from hunter_dqn.hunter_policy import HunterPolicy
from prey_dqn.MultiPreyEnv import MultiPreyEnv
from prey_dqn.prey_model import DQNModelPrey
from prey_dqn.prey_policy import PreyPolicy


def env_creator(env_config):
    return MultiAgentSimEnv(env_config)


if __name__ == "__main__":
    ray.init()
    ModelCatalog.register_custom_model("DQNModel", DQNModel)

    config = {
        "num_hunters": 20,
        "num_preys": 100, }

    env = register_env("MultiAgentSimEnv-v0", env_creator)

    singleAgentEnv = MultiAgentSimEnv(config)
    policies = {"hunter": (HunterPolicy,
                           singleAgentEnv.observation_space_hunter,
                           singleAgentEnv.action_space_hunter,
                           config),
                "prey": (PreyPolicy,
                         singleAgentEnv.observation_space_prey,
                         singleAgentEnv.action_space_prey,
                         config)}


    def policy_mapping_fn(agent_id):
        if "hunter" in agent_id:
            return "hunter"
        else:
            return "prey"


    trainer = DQNTrainer(
        env="MultiAgentSimEnv-v0",
        config={
            "multiagent": {
                "policy_mapping_fn": policy_mapping_fn,
                "policies": policies,
                "policies_to_train": policies
            },
            # disable filters, otherwise we would need to synchronize those
            # as well to the DQN agent
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "framework": "torch",
        }
    )

    tune.run(
        DQNTrainer,
        checkpoint_at_end=True,
        stop={"timesteps_total": 2000},
        config={
            "num_gpus": 0,
            "num_workers": 1,
            "env": "MultiAgentSimEnv-v0",
            "framework": "torch",
            # "sample_batch_size": 50,
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
                "custom_model": "DQNModel",
                "custom_model_config": {
                    # "network_size": [32, 64, 32],
                },
                # extra options to pass to your model
            },
        }
    )
