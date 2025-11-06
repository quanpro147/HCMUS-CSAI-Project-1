from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import time
import numpy as np


class BaseOptimizer(ABC):
    """
    Abstract base class cho tất cả các thuật toán tối ưu.
    """
    
    def __init__(self, name: str = "Base Optimizer"):
        self.name = name
        self.best_solution = None
        self.best_fitness = float('inf')
        self.convergence_curve = []
        self.execution_time = 0
        self.iterations_done = 0
        self.function_evaluations = 0
        
    @abstractmethod
    def optimize(self, problem, max_iter: int = 100, **kwargs) -> Tuple[Any, float]:
        """
        Thực hiện tối ưu hóa.
        
        Args:
            problem: Bài toán cần tối ưu (có method evaluate())
            max_iter: Số iteration tối đa
            **kwargs: Các tham số bổ sung
            
        Returns:
            Tuple[solution, fitness]: Nghiệm tốt nhất và giá trị fitness
        """
        pass
    
    def run(self, problem, max_iter: int = 100, **kwargs) -> Dict[str, Any]:
        """
        Chạy thuật toán và thu thập metrics đầy đủ.
        
        Args:
            problem: Bài toán cần tối ưu
            max_iter: Số iteration tối đa
            **kwargs: Các tham số bổ sung
            
        Returns:
            Dict chứa kết quả và metrics
        """
        # Reset trước khi chạy
        self.reset()
        
        start_time = time.time()
        # Chạy thuật toán
        solution, fitness = self.optimize(problem, max_iter=max_iter, **kwargs)
        end_time = time.time()
        self.execution_time = end_time - start_time
        # Lưu kết quả
        self.best_solution = solution
        self.best_fitness = fitness
        
        # Trả về kết quả đầy đủ
        return {
            'algorithm': self.name,
            'solution': solution,
            'fitness': fitness,
            'execution_time': self.execution_time,
            'iterations': self.iterations_done,
            'function_evaluations': self.function_evaluations,
            'convergence_curve': self.convergence_curve.copy()
        }
    
    def reset(self):
        """Reset trạng thái của thuật toán."""
        self.best_solution = None
        self.best_fitness = float('inf')
        self.convergence_curve = []
        self.execution_time = 0
        self.iterations_done = 0
        self.function_evaluations = 0
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê về quá trình tối ưu.
        """
        return {
            'best_fitness': self.best_fitness,
            'execution_time': self.execution_time,
            'iterations': self.iterations_done,
            'function_evaluations': self.function_evaluations,
            'convergence_rate': self._calculate_convergence_rate(),
            'improvement': self._calculate_improvement()
        }
    
    def _calculate_convergence_rate(self) -> float:
        """
        Tính tốc độ hội tụ của thuật toán.
        """
        if len(self.convergence_curve) < 2:
            return 0.0
        
        initial_fitness = self.convergence_curve[0]
        final_fitness = self.convergence_curve[-1]
        
        if initial_fitness == final_fitness:
            return 1.0
        
        if initial_fitness == 0:
            return 0.0
            
        improvement = abs(initial_fitness - final_fitness)
        return min(1.0, improvement / abs(initial_fitness))
    
    def _calculate_improvement(self) -> float:
        """
        Tính mức cải thiện so với ban đầu.
        """
        if len(self.convergence_curve) < 2:
            return 0.0
        
        initial = self.convergence_curve[0]
        final = self.convergence_curve[-1]
        
        if initial == 0:
            return 0.0
        
        return ((initial - final) / abs(initial)) * 100
    
    def _update_convergence(self):
        """Helper method để update convergence curve."""
        self.convergence_curve.append(self.best_fitness)
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


class SwarmOptimizer(BaseOptimizer):
    """
    Base class cho các thuật toán bầy đàn (Swarm Intelligence).
    """
    
    def __init__(self, name: str, population_size: int = 30):
        """
        Args:
            name: Tên thuật toán
            population_size: Kích thước quần thể/bầy đàn
        """
        super().__init__(name)
        self.population_size = population_size
        self.population = None  # Quần thể hiện tại
        self.fitness_values = None  # Fitness của từng cá thể
    
    def _initialize_population(self, problem, population_size: int = None):
        """
        Khởi tạo quần thể ngẫu nhiên.
        
        Args:
            problem: Bài toán
            population_size: Kích thước quần thể (nếu khác mặc định)
        """
        if population_size is None:
            population_size = self.population_size
        
        # Tạo quần thể ngẫu nhiên
        self.population = np.array([problem.random_solution() 
                                    for _ in range(population_size)])
        
        # Đánh giá fitness
        self.fitness_values = np.array([problem.evaluate(ind) 
                                        for ind in self.population])
        
        self.function_evaluations += population_size
        
        # Update best
        best_idx = np.argmin(self.fitness_values)
        self.best_fitness = self.fitness_values[best_idx]
        self.best_solution = self.population[best_idx].copy()


class TraditionalOptimizer(BaseOptimizer):
    """
    Base class cho các thuật toán tìm kiếm truyền thống.
    """
    
    def __init__(self, name: str):
        """
        Args:
            name: Tên thuật toán
        """
        super().__init__(name)
        self.search_path = []
    
    def get_search_path(self) -> List:
        """
        Lấy đường đi tìm kiếm.
        
        Returns:
            List các trạng thái đã explore
        """
        return self.search_path.copy()
