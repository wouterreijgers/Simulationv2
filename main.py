
from simulation.environment import Environment

if __name__ == '__main__':
    # Define environment
    environment = Environment()
    environment.create_hunters()
    environment.create_preys()
    environment.simulate()


