from base_optimizer import SwarmOptimizer
import numpy as np

class FireflyAlgorithm(SwarmOptimizer):

    def __init__(self):
        super().__init__(name="Firefly Algorithm")
        self.n_fireflies = 20
        self.n_generations = 100
        self.alpha = 0.5 

    def optimize(self):
        pass