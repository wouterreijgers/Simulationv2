import random

from agents.agent import Agent
from util.config_reader import ConfigReader


class Hunter(Agent):
    def __init__(self):
        super().__init__()
        self.energy_to_reproduce = int(ConfigReader("hunter.energy_to_reproduce"))
        self.energy_per_prey_eaten = int(ConfigReader("hunter.energy_per_prey_eaten"))
        self.energy_level = 3*self.energy_per_prey_eaten
        self.max_age = int(ConfigReader("hunter.max_age"))
        self.movements = [[1, 0], [0, 1], [-1, 0], [1, 0], [0, 0]]

    def step(self):
        reward = self.energy_level
        super(Hunter, self).step()
        self.energy_level = self.energy_level - 1
        reward = reward - self.energy_level
        return self.move(), reward, self.energy_level, self.age

    def move(self):
        """
        This is the move function, there are 5 actions. The animal kan go up, down, left, right or stand still.
        :return:
        """
        action = self.movements[random.randint(0, 4)]
        if int(ConfigReader("width"))>self.x_pos + action[0]>0:
            self.x_pos = self.x_pos + action[0]
        if int(ConfigReader("height"))>self.y_pos + action[0]>0:
            self.y_pos = self.y_pos + action[1]
        print("new position: ", self.x_pos, self.y_pos)
        return action;

    def give_birth(self):
        """
        The hunter can choose to reproduce or not. this is only possible if it has enough energy.
        :return:
        """
        if self.energy_level > self.energy_to_reproduce:
            birth_probability = random.randint(0, 100)
            if birth_probability > 0:  # 50% chance the agent will reproduce if he has the energy for it.
                self.energy_level = self.energy_level-self.energy_to_reproduce
                return True
        return False

    def hunt(self, prey):
        """
        The hunter checks if there is a prey in its region
        :return:
        """
        x, y = prey.getPosition()
        if self.x_pos-1<=x<=self.x_pos+1 and self.y_pos-1<=y<=self.y_pos+1:
            print("hunted")
            self.energy_level = self.energy_level + self.energy_per_prey_eaten
            return True
        return False

    def print_vitals(self):
        print("HUNTER VITALS: age ", self.age, " energy ", self.energy_level, " pos ", self.y_pos, self.x_pos)

    def alive(self):
        if self.energy_level > 0 and self.age < self.max_age:
            return True
        return False
