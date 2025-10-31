from ..base_optimizer import TraditionalOptimizer
import numpy as np
from typing import Any, Tuple


class HillClimbing(TraditionalOptimizer):
    """
    Hill Climbing Algorithm implementation.
    """
    
    def __init__(self, max_neighbors=None, step_size=None, name: str = "Hill Climbing"):
        """
        Args:
            max_neighbors: Số lượng neighbors tối đa để thử mỗi iteration
            step_size: Kích thước bước di chuyển khi tạo neighbor
            name: Tên của thuật toán
        """
        super().__init__(name)
        self.max_neighbors = max_neighbors
        self.step_size = step_size if step_size is not None else 0.1
        
    def optimize(self, problem, max_iter: int = 100, **kwargs) -> Tuple[Any, float]:
        """
        Thực hiện tối ưu hóa bằng Hill Climbing.
        
        Args:
            problem: Bài toán cần tối ưu (có method evaluate())
            max_iter: Số iteration tối đa
            **kwargs: Các tham số bổ sung
            
        Returns:
            Tuple[solution, fitness]: Nghiệm tốt nhất và giá trị fitness
        """
        # Khởi tạo điểm bắt đầu ngẫu nhiên
        current = problem.random_solution()
        current_fitness = problem.evaluate(current)
        self.function_evaluations += 1
        
        # Cập nhật best hiện tại
        self.best_solution = current.copy()
        self.best_fitness = current_fitness
        self._update_convergence()
        
        # Thêm trạng thái đầu tiên vào search path
        self.search_path.append(current.copy())
        
        # Lặp cho đến khi hết số iterations
        for iteration in range(max_iter):
            self.iterations_done += 1
            
            # Sinh neighbor bằng cách thêm nhiễu ngẫu nhiên
            neighbor = current + np.random.uniform(-self.step_size, 
                                                self.step_size, 
                                                size=current.shape)
            
            # Đảm bảo neighbor nằm trong bounds
            if hasattr(problem, 'bounds'):
                lb, ub = problem.bounds
                neighbor = np.clip(neighbor, lb, ub)
            
            # Đánh giá neighbor
            neighbor_fitness = problem.evaluate(neighbor)
            self.function_evaluations += 1
            
            # Chấp nhận neighbor nếu tốt hơn
            if neighbor_fitness < current_fitness:
                current = neighbor.copy()
                current_fitness = neighbor_fitness
                
                
                if current_fitness < self.best_fitness:
                    self.best_solution = current.copy()
                    self.best_fitness = current_fitness
                
           
            self.search_path.append(current.copy())
            
            
            self._update_convergence()
            
        return self.best_solution, self.best_fitness