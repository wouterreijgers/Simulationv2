import os


def ConfigReader(config):
    file = open("/home/wouter/PycharmProjects/I-DistributedAI/Simulation/simulation/simulation_param")
    for conf in file:
        configuration = conf.split(" = ")
        if configuration[0] == config:
            return configuration[1]
    return False