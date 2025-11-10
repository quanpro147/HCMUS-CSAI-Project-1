import numpy as np
from .base_problem import ContinuousProblem


class SphereFunction(ContinuousProblem):
    def __init__(self, dim=10, bounds=(-5.12, 5.12), shift=False, rotate=False, seed=None):
        super().__init__("Sphere Function", dim, bounds, shift, rotate, seed)
        self.optimal_value = 0.0

    def evaluate(self, x: np.ndarray) -> float:
        z = self.transform(x)
        return np.sum(z ** 2)


class RastriginFunction(ContinuousProblem):
    def __init__(self, dim=10, bounds=(-5.12, 5.12), shift=False, rotate=False, seed=None):
        super().__init__("Rastrigin Function", dim, bounds, shift, rotate, seed)
        self.optimal_value = 0.0

    def evaluate(self, x: np.ndarray) -> float:
        z = self.transform(x)
        A = 10
        return A * self.dim + np.sum(z ** 2 - A * np.cos(2 * np.pi * z))


class AckleyFunction(ContinuousProblem):
    def __init__(self, dim=10, bounds=(-32.768, 32.768), shift=False, rotate=False, seed=None):
        super().__init__("Ackley Function", dim, bounds, shift, rotate, seed)
        self.optimal_value = 0.0

    def evaluate(self, x: np.ndarray) -> float:
        z = self.transform(x)
        a, b, c = 20, 0.2, 2 * np.pi
        n = self.dim
        sum1 = np.sum(z ** 2)
        sum2 = np.sum(np.cos(c * z))
        term1 = -a * np.exp(-b * np.sqrt(sum1 / n))
        term2 = -np.exp(sum2 / n)
        return term1 + term2 + a + np.e
