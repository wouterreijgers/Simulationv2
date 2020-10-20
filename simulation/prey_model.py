import random

from agents.prey import Prey


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

