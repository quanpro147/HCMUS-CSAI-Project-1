from ..base_optimizer import TraditionalOptimizer
import numpy as np
from typing import Any, Tuple


class SimulatedAnnealing(TraditionalOptimizer):
    """
    Simulated Annealing Algorithm implementation.
    
    SA là thuật toán tối ưu lấy cảm hứng từ quá trình làm nguội kim loại.
    Khác với Hill Climbing, SA có thể chấp nhận nghiệm xấu hơn với xác suất 
    giảm dần theo thời gian (nhiệt độ).
    """
    
    def __init__(self, initial_temp=None, cooling_rate=None, 
                 min_temp=None, name: str = "Simulated Annealing"):
        """
        Args:
            initial_temp: Nhiệt độ ban đầu (T0)
            cooling_rate: Tỷ lệ làm nguội (alpha), thường 0.95-0.99
            min_temp: Nhiệt độ tối thiểu để dừng
            name: Tên của thuật toán
        """
        from config import ALGORITHM_PARAMS
        
        # Lấy tham số từ config nếu không được truyền vào
        sa_params = ALGORITHM_PARAMS.get('simulated_annealing', {})
        initial_temp = initial_temp if initial_temp is not None else sa_params.get('initial_temp', 1000)
        cooling_rate = cooling_rate if cooling_rate is not None else sa_params.get('cooling_rate', 0.95)
        min_temp = min_temp if min_temp is not None else sa_params.get('min_temp', 1e-3)
        
        super().__init__(name)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        
    def optimize(self, problem, max_iter: int = 1000, **kwargs) -> Tuple[Any, float]:
        """
        Thực hiện tối ưu hóa bằng Simulated Annealing.
        
        Args:
            problem: Bài toán cần tối ưu (có method evaluate(), random_solution(), get_neighbors())
            max_iter: Số iteration tối đa
            **kwargs: Tham số bổ sung
            
        Returns:
            Tuple[solution, fitness]: Nghiệm tốt nhất và giá trị fitness
        """
        # Khởi tạo nghiệm ban đầu
        current = problem.random_solution()
        current_fitness = problem.evaluate(current)
        self.function_evaluations += 1
        
        # Lưu nghiệm tốt nhất
        self.best_solution = current.copy()
        self.best_fitness = current_fitness
        self._update_convergence()
        
        # Thêm vào search path
        self.search_path.append(current.copy())
        
        # Nhiệt độ hiện tại
        temperature = self.initial_temp
        
        # Vòng lặp chính
        iteration = 0
        while temperature > self.min_temp and iteration < max_iter:
            iteration += 1
            self.iterations_done += 1
            
            # Sinh neighbor ngẫu nhiên
            # Nếu problem có get_neighbors, chọn 1 neighbor ngẫu nhiên
            if hasattr(problem, 'get_neighbors'):
                # Lấy tất cả neighbors (có thể tốn kém với TSP lớn)
                # Để tối ưu, chỉ sinh 1 neighbor ngẫu nhiên
                neighbors = self._get_random_neighbor(problem, current)
            else:
                # Fallback: thêm nhiễu ngẫu nhiên (cho continuous problems)
                neighbors = current + np.random.uniform(-0.1, 0.1, size=current.shape)
            
            neighbor = neighbors
            neighbor_fitness = problem.evaluate(neighbor)
            self.function_evaluations += 1
            
            # Tính delta (sự khác biệt fitness)
            delta = neighbor_fitness - current_fitness
            
            # Quyết định chấp nhận neighbor
            if delta < 0:
                # Neighbor tốt hơn -> chấp nhận
                accept = True
            else:
                # Neighbor xấu hơn -> chấp nhận với xác suất e^(-delta/T)
                probability = np.exp(-delta / temperature)
                accept = np.random.random() < probability
            
            if accept:
                current = neighbor.copy()
                current_fitness = neighbor_fitness
                self.search_path.append(current.copy())
                
                # Cập nhật best nếu tốt hơn
                if current_fitness < self.best_fitness:
                    self.best_solution = current.copy()
                    self.best_fitness = current_fitness
            
            # Làm nguội (cooling)
            temperature *= self.cooling_rate
            
            # Cập nhật convergence curve
            self._update_convergence()
        
        return self.best_solution, self.best_fitness
    
    def _get_random_neighbor(self, problem, current):
        """
        Sinh một neighbor ngẫu nhiên.
        
        Với TSP: dùng swap hoặc 2-opt
        Với Knapsack: flip một bit ngẫu nhiên
        """
        # Kiểm tra kiểu bài toán
        problem_type = problem.__class__.__name__
        
        if 'TSP' in problem_type or 'Traveling' in problem_type:
            # TSP: swap 2 thành phố ngẫu nhiên
            neighbor = current.copy()
            i, j = np.random.choice(len(current), size=2, replace=False)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            return neighbor
            
        elif 'Knapsack' in problem_type:
            # Knapsack: flip 1 bit ngẫu nhiên
            neighbor = current.copy()
            idx = np.random.randint(len(current))
            neighbor[idx] = 1 - neighbor[idx]
            return neighbor
            
        else:
            # Generic: lấy 1 neighbor ngẫu nhiên từ danh sách
            neighbors = problem.get_neighbors(current)
            if len(neighbors) > 0:
                return neighbors[np.random.randint(len(neighbors))]
            return current.copy()
