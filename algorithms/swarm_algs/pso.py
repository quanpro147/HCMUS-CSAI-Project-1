# Khoa
from ..base_optimizer import SwarmOptimizer
import numpy as np

class ParticleSwarmOptimization(SwarmOptimizer):
    def __init__(self, population_size=None, w=None, c1=None, c2=None):
        super().__init__(name="Particle Swarm Optimization")
        self.population_size = population_size
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def optimize(self, problem, max_iter=100, **kwargs):
        # TODO: Implement PSO algorithm
        pass