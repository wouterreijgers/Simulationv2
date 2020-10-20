import time

from simulation.hunter_model import HunterModel
from simulation.prey_model import PreyModel
from simulation.simulator import Simulator
from util.config_reader import ConfigReader


class Environment():
    def __init__(self):
        print("INITIALISING ENVIRONMENT:")
        self.width = ConfigReader("width")
        self.height = ConfigReader("height")
        self.start_amount_of_hunter = ConfigReader("start_amount_of_hunter")
        self.start_amount_of_preys = ConfigReader("start_amount_of_prey")
        print("\tWIDTH: ", int(self.width))
        print("\tHEIGHT: ", int(self.height))
        print("\tHUNTERS: ", int(self.start_amount_of_hunter))
        print("\tPREYS: ", int(self.start_amount_of_preys), "\n")

        self.hunter_model = HunterModel()
        self.prey_model = PreyModel()

        self.simulator = Simulator()
        self.simulation_time = 0

    def create_hunters(self):
        for i in range(int(self.start_amount_of_hunter)):
            self.hunter_model.new_hunter()

    def create_preys(self):
        for i in range(int(self.start_amount_of_preys)):
            self.prey_model.new_prey()

    def simulate(self):
        running = True
        while running:
            self.hunter_model.update_hunters()
            self.prey_model.update_preys()
            self.hunter_model.hunt(self.prey_model.get_preys())
            if not self.simulator.run(self.simulation_time, self.hunter_model.get_hunters(), self.prey_model.get_preys()):
                running = False
            if not self.hunter_model.has_hunters():
                running = False
            time.sleep(1)
            self.simulation_time += 1
        self.simulator.quit()
        print("SIMULATION ENDED")
