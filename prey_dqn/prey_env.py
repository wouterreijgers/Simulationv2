import math
import random
import queue

from prey_dqn.prey import Prey
from util.config_reader import ConfigReader


class PreyModel():
    def __init__(self):
        self.preys = []

    def update_preys(self):
        i = len(self.preys)
        living_preys = []
        for id in range(0, i):
            prey = self.preys.pop()
            prey.step()
            if prey.give_birth():
                living_preys.append(Prey())
            if prey.alive():
                living_preys.append(prey)
            #prey.print_vitals()
        self.preys = living_preys

    def has_preys(self):
        if len(self.preys)>0:
            return True
        return False

    def new_prey(self):
        self.preys.append(Prey())

    def get_preys(self):
        return self.preys

    def get_rel_x_y(self, state):
        best_dist = math.inf
        rel_x = 0
        rel_y = 0
        nearest_prey = None
        for prey in self.get_preys():
            x, y = prey.get_position()
            dist = abs(state[0]-x) + abs(state[1]-y)
            #print(dist)
            if dist < best_dist:
                best_dist = dist
                rel_x = x-state[0]
                rel_y = y-state[1]
                nearest_prey = prey
        if best_dist < 5:
            nearest_prey.kill()
            print("kill")
        return rel_x, rel_y

