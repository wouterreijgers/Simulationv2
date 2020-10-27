import pygame

from util.config_reader import ConfigReader
from util.monitoring import Monitoring


class Simulator():
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode([750, 750])
        self.x_unit = 750 / int(ConfigReader("width"))
        self.y_unit = 750 / int(ConfigReader("height"))
        self.monitor = Monitoring()

    def run(self, simulation_time, hunters, preys):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        self.screen.fill((255, 255, 255))
        for observation in hunters:
            x, y = observation
            #print(x.item(), y, observation)
            surf = pygame.Surface((self.x_unit, self.y_unit))
            surf.fill((255, 127, 80))
            self.screen.blit(surf, (x.item() * self.x_unit, y.item() * self.y_unit))
        for prey in preys:
            x, y = prey.get_position()
            surf = pygame.Surface((self.x_unit, self.y_unit))
            surf.fill((220, 220, 220))
            self.screen.blit(surf, (x * self.x_unit, y * self.y_unit))
        pygame.display.flip()
        self.monitor.log(simulation_time, len(hunters), len(preys))
        return True

    def quit(self):
        pygame.quit()
        self.monitor.plot()
