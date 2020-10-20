import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class Monitoring():
    def __init__(self):
        self.time = []
        self.hunter_population = []
        self.prey_population = []

    def log(self, time, amount_of_hunters, amount_of_preys):
        self.time.append(time)
        self.hunter_population.append(amount_of_hunters)
        self.prey_population.append(amount_of_preys)

    def plot(self):
        plt.plot(self.time, self.hunter_population, 'r', self.time, self.prey_population, 'g')
        red_patch = mpatches.Patch(color='red', label='The hunter population')
        green_patch = mpatches.Patch(color='green', label='The prey population')
        plt.legend(handles=[red_patch, green_patch])
        plt.xlabel("Time")
        plt.ylabel("Population")
        plt.show()
