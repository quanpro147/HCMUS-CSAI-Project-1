from abc import ABC
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
    def __init__(self, prob_name, dim, bounds: Tuple[float, float],
                 shift=False, rotate=False, seed=None):
        super().__init__(prob_name)
        self.dim = dim
        self.lower_bound = bounds[0]
        self.upper_bound = bounds[1]

        np.random.seed(seed)
        self.shift_enabled = shift
        self.rotate_enabled = rotate

        # Dịch chuyển (shift) nghiệm tối ưu
        if shift:
            self.shift_vector = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        else:
            self.shift_vector = np.zeros(self.dim)

        # Thêm ma trận xoay hệ trục toạ độ
        if rotate:
            A = np.random.randn(self.dim, self.dim)
            Q, _ = np.linalg.qr(A)
            self.rotation_matrix = Q
        else:
            self.rotation_matrix = np.eye(self.dim)

    def get_bounds(self):
        lower = np.full(self.dim, self.lower_bound)
        upper = np.full(self.dim, self.upper_bound)
        return lower, upper

    def random_solution(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

    def clip_solution(self, solution):
        return np.clip(solution, self.lower_bound, self.upper_bound)

    def is_valid(self, solution):
        return np.all(solution >= self.lower_bound) and np.all(solution <= self.upper_bound)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Áp dụng phép biến đổi (shift + rotation) trước khi evaluate.
        """
        z = x - self.shift_vector
        z = self.rotation_matrix @ z
        return z

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