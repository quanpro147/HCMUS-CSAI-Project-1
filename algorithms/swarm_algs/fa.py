from algorithms.base_optimizer import SwarmOptimizer
import numpy as np

class FireflyAlgorithm(SwarmOptimizer):

    def __init__(self, population_size=30, beta0=1, gamma=1, alpha=0.2):
        """
        Firefly Algorithm.
        
        Args:
            population_size: Số lượng fireflies
            beta0: Attractiveness tại r=0
            gamma: Light absorption coefficient
            alpha: Randomization parameter
        """
        super().__init__(name="Firefly Algorithm", population_size=population_size)
        self.beta0 = beta0
        self.gamma = gamma
        self.alpha = alpha
        self.history = []
        
    def _distance(self, x1, x2):
        return np.linalg.norm(x1 - x2)
    
    def optimize(self, problem, max_iter=100, **kwargs):
        """
        Chạy Firefly Algorithm.
        
        Args:
            problem: Bài toán cần tối ưu
            max_iter: Số iteration tối đa
            
        Returns:
            Tuple[solution, fitness]: Nghiệm tốt nhất và fitness
        """
        # Khởi tạo population
        self._initialize_population(problem, self.population_size)
        
        # Get bounds
        lb, ub = problem.get_bounds()
        dim = problem.dim

        for iteration in range(max_iter):
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if self.fitness_values[j] < self.fitness_values[i]:
                        r = self._distance(self.population[i], self.population[j])
                        beta = self.beta0 * np.exp(-self.gamma * r ** 2)
                        step = beta * (self.population[j] - self.population[i])
                        random_walk = self.alpha * (np.random.rand(dim) - 0.5)
                        self.population[i] += step + random_walk
                        self.population[i] = np.clip(self.population[i], lb, ub)
                        self.fitness_values[i] = problem.evaluate(self.population[i])
                        self.function_evaluations += 1

            # Update best
            best_idx = np.argmin(self.fitness_values)
            if self.fitness_values[best_idx] < self.best_fitness:
                self.best_fitness = self.fitness_values[best_idx]
                self.best_solution = self.population[best_idx].copy()
            
            # Save history và convergence
            self.history.append(self.population.copy())
            self._update_convergence()
            self.iterations_done += 1

        return self.best_solution, self.best_fitness
    
    def get_history(self):
        return self.history