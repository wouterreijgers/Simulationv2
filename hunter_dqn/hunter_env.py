import random
import pandas as pd
from dataclasses import dataclass

import gym
from gym import spaces, logger
import numpy as np

from util.config_reader import ConfigReader


class HunterEnv(gym.Env):
    """
    Description:
        The hunter will try to catch preys in order to survive, as soon as the hunter collected enough energy
        it can reproduce and make sure the species survives.

    Observation:
        Type: Box(4)
        Num     Observation               Min                Max
        0       Age                       0                  max_age defined in the simulation_param file
        1       Energy level              0                  100
        2       rel x to closest prey     0                  width defined in the simulation_param file
        3       rel y to closest prey     0                  Height defined in the simulation_param file

    Actions:
        Type: Discrete(5)
        Num   Action
        0     Reproduce, this is only possible if he has enough energy.
        1     Move up
        2     Move right
        3     Move down
        4     Move left
        Note: The hunter can not move out of the field defined in the simulation parameters, it can still perform the
        actions but it would result in no movement.

    Starting state:
        The hunter will start with age 0 and an energy level three times the amount of energy it receives
        from eating a prey. The other parameters will be defined randomly.

    Reward:
        to be decided

    Termination:
        The hunter dies when his energy is zero or when his age reaches the maximum age.
    """

    def __init__(self, preys):
        # Static configurations
        self.max_age = int(ConfigReader("hunter.max_age"))
        self.max_energy = 100
        self.width = int(ConfigReader("width"))
        self.height = int(ConfigReader("height"))
        high = np.array([self.max_age, self.max_energy, self.width, self.height], dtype=np.float32)
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(np.array([0, 0, 0, 0]), high, dtype=np.float32)
        self.energy_to_reproduce = int(ConfigReader("hunter.energy_to_reproduce"))
        self.energy_per_prey_eaten = int(ConfigReader("hunter.energy_per_prey_eaten"))
        self.preys = preys
        # Hunter specific
        self.age = 0
        self.energy = 3 * self.energy_per_prey_eaten
        self.x = random.randint(0, self.width)
        self.y = random.randint(0, self.height)

        self.state = None
        self.steps_beyond_done = None

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        age, energy, x_to_prey, y_to_prey = self.state
        reproduce = False

        # cost of living
        age += 1
        energy -= 1

        # perform the action
        #if action == 0 and self.energy >= self.energy_to_reproduce:
        if energy >= self.energy_to_reproduce:
            energy -= self.energy_to_reproduce
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
        x_to_prey, y_to_prey = self.preys.get_rel_x_y([self.x, self.y])
        if(abs(x_to_prey) + abs(y_to_prey)) < 3:
            print("hunt")
            energy += self.energy_per_prey_eaten

        self.state = (age, energy, x_to_prey, y_to_prey)
        done = bool(
            age > self.max_age
            or energy <= 0
        )

        # TODO: Define rewards when the hunter catches a prey/gives birth/...

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
        self.state = (0, self.energy, random.randint(0, self.width), random.randint(0, self.width))
        self.steps_beyond_done = None
        return np.array(self.state)

    def get_position(self):
        return np.array([self.x, self.y])

    def render(self, mode='human'):
        pass
