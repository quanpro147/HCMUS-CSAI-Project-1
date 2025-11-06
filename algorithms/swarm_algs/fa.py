from ..base_optimizer import SwarmOptimizer
import numpy as np
from config import ALGORITHM_PARAMS

class FireflyAlgorithm(SwarmOptimizer):

    def __init__(self, population_size=None, beta0=None, gamma=None, alpha=None):
        # Lấy tham số từ config nếu không được truyền vào
        fa_params = ALGORITHM_PARAMS.get('fa', {})
        population_size = population_size if population_size is not None else fa_params.get('population_size', 30)
        beta0 = beta0 if beta0 is not None else fa_params.get('beta0', 1.0)
        gamma = gamma if gamma is not None else fa_params.get('gamma', 1.0)
        alpha = alpha if alpha is not None else fa_params.get('alpha', 0.2)
        
        super().__init__(name="Firefly Algorithm", population_size=population_size)
        self.beta0 = beta0
        self.gamma = gamma
        self.alpha = alpha
        self.history = []

    def optimize(self, problem, max_iter=100, **kwargs):
        self._initialize_population(problem, self.population_size)
        lb, ub = problem.get_bounds()
        dim = problem.dim

        for iteration in range(max_iter):
            # === Compute pairwise distances ===
            diff = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
            distances = np.linalg.norm(diff, axis=2)  # shape (n, n)

            # === Compute attractiveness matrix ===
            beta_matrix = self.beta0 * np.exp(-self.gamma * distances**2)

            # === Firefly movement ===
            for i in range(self.population_size):
                # Find which fireflies are brighter (better)
                better_mask = self.fitness_values < self.fitness_values[i]
                if np.any(better_mask):
                    # Vectorized movement toward all better fireflies
                    attraction = np.sum(
                        beta_matrix[i, better_mask, np.newaxis]
                        * (self.population[better_mask] - self.population[i]),
                        axis=0
                    )
                    # Random walk
                    random_walk = self.alpha * (np.random.rand(dim) - 0.5)
                    self.population[i] += attraction + random_walk

            # === Boundary handling ===
            self.population = np.clip(self.population, lb, ub)

            # === Evaluate ===
            self.fitness_values = np.array([problem.evaluate(ind) for ind in self.population])
            self.function_evaluations += self.population_size

            # === Update best ===
            best_idx = np.argmin(self.fitness_values)
            if self.fitness_values[best_idx] < self.best_fitness:
                self.best_fitness = self.fitness_values[best_idx]
                self.best_solution = self.population[best_idx].copy()

            self.history.append(self.population.copy())
            self._update_convergence()
            self.iterations_done += 1

        return self.best_solution, self.best_fitness

    def get_history(self):
        return self.history
