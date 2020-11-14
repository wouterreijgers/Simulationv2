import gym

gym.envs.register(
    id='MultiHunterEnv-v0',
    entry_point='hunter_dqn.MultiHunterEnv:MultiHunterEnv',
    kwargs={'config': {"num_agents": 20}, 'preys': {}}
)