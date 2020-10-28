import ray

from simulation.environment import Environment

if __name__ == '__main__':
    ray.init()

    # Define environment
    environment = Environment()
    environment.create_hunters()
    environment.create_preys()
    environment.simulate()


