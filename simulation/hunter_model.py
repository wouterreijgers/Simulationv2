import random

from agents.hunter import Hunter


class HunterModel():
    def __init__(self):
        self.hunters = []

    def update_hunters(self):
        i = len(self.hunters)
        living_hunters = []
        for id in range(0, i):
            hunter = self.hunters.pop()
            hunter.step()
            if hunter.give_birth():
                living_hunters.append(Hunter())
            if hunter.alive():
                living_hunters.append(hunter)
            hunter.print_vitals()
        self.hunters = living_hunters

    def has_hunters(self):
        if len(self.hunters)>0:
            return True
        return False

    def new_hunter(self):
        self.hunters.append(Hunter())

    def get_hunters(self):
        return self.hunters

    def hunt(self, preys):
        for hunter in self.hunters:
            for prey in preys:
                if hunter.hunt(prey):
                    prey.kill()


