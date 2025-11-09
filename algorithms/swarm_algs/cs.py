import numpy as np
import math
from ..base_optimizer import SwarmOptimizer
from config import ALGORITHM_PARAMS


class CuckooSearch(SwarmOptimizer):

    def __init__(self, population_size=None, pa=None, alpha=None):

        cs_params = ALGORITHM_PARAMS.get('cs', {})
        population_size = population_size if population_size is not None else cs_params.get('population_size', 25)
        pa = pa if pa is not None else cs_params.get('pa', 0.25)
        alpha = alpha if alpha is not None else cs_params.get('alpha', 0.01)
        
        super().__init__(name="Cuckoo Search", population_size=population_size)
        self.pa = pa
        self.alpha = alpha

    def _levy_flight(self, size, beta=1.5):
        sigma_u = (
            (math.gamma(1 + beta) * np.sin(np.pi * beta / 2))
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)

        u = np.random.normal(0, sigma_u, size)
        v = np.random.normal(0, 1, size)
        return u / (np.abs(v) ** (1 / beta))

    def optimize(self, problem, max_iter=100, **kwargs):
        self._initialize_population(problem)
        n = self.population_size
        dim = self.population.shape[1]

        for t in range(max_iter):
            # === Levy flight vectorized ===
            steps = self._levy_flight((n, dim))
            new_population = self.population + self.alpha * steps * (self.population - self.best_solution)
            new_population = problem.clip_solution(new_population)

            # Evaluate all new solutions at once
            new_fitness = np.array([problem.evaluate(sol) for sol in new_population])
            self.function_evaluations += n

            # Replace if better
            improved = new_fitness < self.fitness_values
            self.population[improved] = new_population[improved]
            self.fitness_values[improved] = new_fitness[improved]

            # === Abandon worst nests ===
            abandon_mask = np.random.rand(n) < self.pa
            if np.any(abandon_mask):
                new_nests = np.array([problem.random_solution() for _ in range(np.sum(abandon_mask))])
                new_fits = np.array([problem.evaluate(sol) for sol in new_nests])
                self.population[abandon_mask] = new_nests
                self.fitness_values[abandon_mask] = new_fits
                self.function_evaluations += np.sum(abandon_mask)

            # === Update global best ===
            best_idx = np.argmin(self.fitness_values)
            if self.fitness_values[best_idx] < self.best_fitness:
                self.best_fitness = self.fitness_values[best_idx]
                self.best_solution = self.population[best_idx].copy()

        return self.best_solution, self.best_fitness
