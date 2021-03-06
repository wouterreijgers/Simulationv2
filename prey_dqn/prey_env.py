import random
import pandas as pd
from dataclasses import dataclass

import gym
from gym import spaces, logger
import numpy as np

from util.config_reader import ConfigReader


class PreyEnv(gym.Env):
    """
    Description:
        The prey must try to avoid the hunters and create a population as big as possible.

    Observation:
        Type: Box(4)
        Num     Observation               Min                Max
        0       Age                       0                  max_age defined in the simulation_param file
        1       rel x to closest hunter   0                  width defined in the simulation_param file
        2       rel y to closest hunter   0                  Height defined in the simulation_param file

    Actions:
        Type: Discrete(5)
        Num   Action
        0     Move up
        1     Move right
        2     Move down
        3     Move left
        Note: The prey can not move out of the field defined in the simulation parameters, it can still perform the
        actions but it would result in no movement.

    Starting state:
        The prey will start with age 0.

    Reward:
        to be decided

    Termination:
        The prey dies when his age is the max_age or when he is eaten by a hunter.
    """

    def __init__(self):
        # Static configurations
        self.max_age = int(ConfigReader("prey.max_age"))
        self.width = int(ConfigReader("width"))
        self.height = int(ConfigReader("height"))
        self.birth_rate = int(ConfigReader("prey.birth_rate"))
        #print("birth_rate", self.birth_rate)
        high = np.array([self.max_age, self.width, self.height], dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(np.array([0, 0, 0]), high, dtype=np.float32)
        # Prey specific
        self.age = 0
        self.x = random.randint(0, self.width)
        self.y = random.randint(0, self.height)

        self.state = None
        self.steps_beyond_done = None

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        age, x_to_prey, y_to_prey = self.state
        reproduce = False

        # cost of living
        age += 1

        # perform the action
        birth_probability = random.randint(0, 100)
        #print(birth_probability)
        if birth_probability <= self.birth_rate:
            #print("reproducs")
            reproduce = True
        if action == 1 and self.y < self.height - 1:
            self.y += 1
        if action == 2 and self.x < self.width - 1:
            self.x += 1
        if action == 3 and self.y > 0:
            self.y -= 1
        if action == 4 and self.x > 0:
            self.x += 1

        # find closest prey and 'eat' if close enough
        x_to_hunter, y_to_hunter = 10, 10#self.hunters.get_rel_x_y([self.x, self.y])
        self.state = (age, x_to_hunter, y_to_hunter)
        done = bool(
            (abs(x_to_hunter) + abs(y_to_hunter)) < 3
            or age > self.max_age
        )

        # TODO: Define rewards when the prey gives birth/...

        if not done:
            reward = 1
        elif self.steps_beyond_done is None:
            # Hunter jus died
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, reproduce

    def reset(self):
        self.state = (0, random.randint(0, self.width), random.randint(0, self.width))
        self.steps_beyond_done = None
        return np.array(self.state)

    def get_position(self):
        return np.array([self.x, self.y])

    def render(self, mode='human'):
        pass


