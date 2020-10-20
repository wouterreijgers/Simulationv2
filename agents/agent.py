import random

from util.config_reader import ConfigReader


class Agent:
    def __init__(self):
        self.x_pos = random.randint(0, int(ConfigReader("width")))
        self.y_pos = random.randint(0, int(ConfigReader("height")))
        self.age = 0

    def step(self):
        self.age += 1

    def getPosition(self):
        return self.x_pos, self.y_pos

