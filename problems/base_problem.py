from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

class BaseProblem(ABC):

    def __init__(self, prob_name):
        self.prob_name = prob_name
        self.optimal_value = None
        self.optimal_solution = None
    
    def evaluate(self, solution):
        pass

    def get_optimal_value(self):
        return self.optimal_value
    
    def __str__(self):
        return self.prob_name
    
    
class ContinuousProblem(BaseProblem):

    def __init__(self, prob_name, dim, bounds: Tuple[float, float]):
        super().__init__(prob_name)
        self.dim = dim
        self.lower_bound = bounds[0]
        self.upper_bound = bounds[1]
    
    def get_bounds(self):
        """
        Lấy biên của không gian tìm kiếm.
        """
        lower = np.full(self.dim, self.lower_bound)
        upper = np.full(self.dim, self.upper_bound)
        return lower, upper
    
    def random_solution(self):
        """
        Tạo nghiệm ngẫu nhiên trong không gian tìm kiếm.
        """
        return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
    
    def clip_solution(self, solution):
        """
        Điều chỉnh nghiệm về trong biên cho phép.
        """
        return np.clip(solution, self.lower_bound, self.upper_bound)
    
    def is_valid(self, solution):
        """
        Kiểm tra nghiệm có thõa điều kiện biên không.
        """
        return np.all(solution >= self.lower_bound) and np.all(solution <= self.upper_bound)

class DiscreteProblem(BaseProblem):

    def __init__(self, prob_name):
        super().__init__(prob_name)

    def random_solution(self):
        """        
        Tạo nghiệm ngẫu nhiên.
        """
        pass

    def is_valid(self, solution):
        """
        Kiểm tra nghiệm có hợp lệ không.
        """
        pass