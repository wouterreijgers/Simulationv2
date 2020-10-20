import random

from agents.agent import Agent
from util.config_reader import ConfigReader


class Prey(Agent):
    def __init__(self):
        super().__init__()
        self.energy_to_reproduce = int(ConfigReader("prey.energy_to_reproduce"))
        self.max_age = int(ConfigReader("prey.max_age"))
        self.birth_rate = int(ConfigReader("prey.birth_rate"))
        self.movements = [[1, 0], [0, 1], [-1, 0], [1, 0], [0, 0]]
        self.hunted = False

    def step(self):
        super(Prey, self).step()
        self.move()
        self.give_birth()

    def move(self):
        # 5 actions, Up, right, down, left, don't move
        action = self.movements[random.randint(0, 4)]
        if int(ConfigReader("width")) > self.x_pos + action[0] > 0:
            self.x_pos = self.x_pos + action[0]
        if int(ConfigReader("height")) > self.y_pos + action[0] > 0:
            self.y_pos = self.y_pos + action[1]
        print("new position: ", self.x_pos, self.y_pos)

    def give_birth(self):
        birth_probability = random.randint(0, 100)
        if birth_probability < self.birth_rate:
            return True
        return False

    def alive(self):
        if self.age < self.max_age and not self.hunted:
            return True
        return False

    def kill(self):
        self.hunted = True