import numpy as np
from .base_problem import ContinuousProblem


class SphereFunction(ContinuousProblem):

    def __init__(self, dim=10):
        super().__init__(prob_name="Sphere Function", dim=dim, bounds=(-5.12, 5.12))
        self.optimal_value = 0.0
        self.optimal_solution = np.zeros(dim)

    def evaluate(self, x: np.ndarray) -> float:
        return np.sum(x ** 2)


class RastriginFunction(ContinuousProblem):

    def __init__(self, dim=10):
        super().__init__(prob_name="Rastrigin Function", dim=dim, bounds=(-5.12, 5.12))
        self.optimal_value = 0.0
        self.optimal_solution = np.zeros(dim)

    def evaluate(self, x: np.ndarray) -> float:
        A = 10
        return A * self.dim + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))

class RosenbrockFunction(ContinuousProblem):
    
    def __init__(self, dim=10):
        super().__init__(prob_name="Rosenbrock Function", dim=dim, bounds=(-5, 10))
        self.optimal_value = 0.0
        self.optimal_solution = np.ones(dim)
    
    def evaluate(self, x: np.ndarray) -> float:
        return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

class AckleyFunction(ContinuousProblem):
    
    def __init__(self, dim=10):
        super().__init__(prob_name="Ackley Function", dim=dim, bounds=(-32.768, 32.768))
        self.optimal_value = 0.0
        self.optimal_solution = np.zeros(dim)
    
    def evaluate(self, x: np.ndarray) -> float:
        a = 20
        b = 0.2
        c = 2 * np.pi
        
        sum1 = np.sum(x ** 2)
        sum2 = np.sum(np.cos(c * x))
        
        term1 = -a * np.exp(-b * np.sqrt(sum1 / self.dim))
        term2 = -np.exp(sum2 / self.dim)
        
        return term1 + term2 + a + np.e