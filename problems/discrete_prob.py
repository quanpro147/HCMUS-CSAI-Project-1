import numpy as np
from typing import List, Tuple
from .base_problem import DiscreteProblem

class TravelingSalesmanProblem(DiscreteProblem):
    """
    Bài toán người du lịch (TSP).
    
    Mỗi nghiệm là một permutation của các thành phố (0 đến n_cities-1).
    Mục tiêu là tìm tour ngắn nhất đi qua tất cả các thành phố đúng một lần.
    """
    
    def __init__(self, name: str = "TSP", n_cities: int = 20, 
                 distance_matrix: np.ndarray = None, seed: int = None):
        """
        Args:
            name: Tên bài toán
            n_cities: Số lượng thành phố
            distance_matrix: Ma trận khoảng cách giữa các thành phố
            seed: Seed cho random
        """
        super().__init__(name)
        self.n_cities = n_cities
        if distance_matrix is not None:
            if distance_matrix.shape != (n_cities, n_cities):
                raise ValueError("Distance matrix phải có kích thước (n_cities, n_cities)")
            self.distance_matrix = distance_matrix
        else:
            self.distance_matrix = self._generate_distance_matrix(seed)
        self.optimal_value = None
        self.coords = None  # Lưu tọa độ các thành phố nếu được tạo ngẫu nhiên

    def _generate_distance_matrix(self, seed: int = None) -> np.ndarray:
        """
        Tạo ma trận khoảng cách ngẫu nhiên.
        
        Args:
            seed: Random seed
            
        Returns:
            np.ndarray: Ma trận khoảng cách
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Tạo tọa độ ngẫu nhiên cho các thành phố
        self.coords = np.random.rand(self.n_cities, 2) * 100
        
        # Tính khoảng cách Euclidean
        distances = np.zeros((self.n_cities, self.n_cities))
        for i in range(self.n_cities):
            for j in range(i + 1, self.n_cities):
                dist = np.linalg.norm(self.coords[i] - self.coords[j])
                distances[i, j] = dist
                distances[j, i] = dist
        return distances
    
    def evaluate(self, solution: np.ndarray) -> float:
        """
        Tính độ dài của tour.
        
        Args:
            solution: Một permutation của các thành phố
            
        Returns:
            float: Tổng độ dài của tour
        """
        total_distance = 0
        for i in range(len(solution)):
            from_city = solution[i]
            to_city = solution[(i + 1) % len(solution)]
            total_distance += self.distance_matrix[from_city, to_city]
        return total_distance

    def random_solution(self) -> np.ndarray:
        """
        Tạo một tour ngẫu nhiên.
        
        Returns:
            np.ndarray: Một permutation ngẫu nhiên của các thành phố
        """
        return np.random.permutation(self.n_cities)
    
    def is_valid(self, solution: np.ndarray) -> bool:
        """
        Kiểm tra tính hợp lệ của tour.
        
        Tour hợp lệ phải:
        - Có độ dài bằng số thành phố
        - Chứa mỗi thành phố đúng một lần
        - Các chỉ số thành phố phải nằm trong khoảng hợp lệ
        
        Args:
            solution: Tour cần kiểm tra
            
        Returns:
            bool: True nếu tour hợp lệ, False nếu không
        """
        solution = np.array(solution)
        return (len(solution) == self.n_cities and
                len(set(solution)) == self.n_cities and
                all(0 <= city < self.n_cities for city in solution))
    
    def swap_cities(self, solution: np.ndarray, i: int, j: int) -> np.ndarray:
        """
        Hoán đổi vị trí hai thành phố trong tour.
        
        Args:
            solution: Tour hiện tại
            i, j: Vị trí hai thành phố cần hoán đổi
            
        Returns:
            np.ndarray: Tour mới sau khi hoán đổi
        """
        new_solution = solution.copy()
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
        return new_solution
    
    def two_opt(self, solution: np.ndarray, i: int, j: int) -> np.ndarray:
        """
        Áp dụng phép biến đổi 2-opt.
        
        Args:
            solution: Tour hiện tại
            i, j: Đoạn cần đảo ngược
            
        Returns:
            np.ndarray: Tour mới sau khi áp dụng 2-opt
        """
        new_solution = solution.copy()
        new_solution[i:j+1] = new_solution[i:j+1][::-1]
        return new_solution
        
    def get_neighbors(self, solution: np.ndarray, method: str = 'swap') -> List[np.ndarray]:
        """
        Sinh các neighbors của một nghiệm.
        
        Args:
            solution: Nghiệm hiện tại
            method: Phương pháp sinh neighbor ('swap' hoặc '2-opt')
            
        Returns:
            List[np.ndarray]: Danh sách các neighbors
        """
        neighbors = []
        n = len(solution)
        
        if method == 'swap':
            for i in range(n):
                for j in range(i+1, n):
                    neighbors.append(self.swap_cities(solution, i, j))
        elif method == '2-opt':
            for i in range(n-1):
                for j in range(i+2, n):
                    neighbors.append(self.two_opt(solution, i, j))
        else:
            raise ValueError("method phải là 'swap' hoặc '2-opt'")
            
        return neighbors
    

class KnapsackProblem(DiscreteProblem):

    def __init__(self, n_items=20, capacity=None, weights=None, values=None, seed=None):
        super().__init__(prob_name="Knapsack Problem")
        self.n_items = n_items
        if seed is not None:
            np.random.seed(seed)
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.random.randint(1, 50, n_items)
        if values is not None:
            self.values = values
        else:
            self.values = np.random.randint(1, 100, n_items)
        if capacity is not None:
            self.capacity = capacity
        else:
            self.capacity = int(0.5 * np.sum(self.weights))
        self.optimal_value = None

    def evaluate(self, solution: np.ndarray) -> float:
        """
            solution: là vector nhị phân với 0 là ko chọn, 1 là chọn
        """
        solution = np.array(solution, dtype=int)
        total_weight = np.sum(solution * self.weights)
        total_value = np.sum(solution * self.values)
        if total_weight > self.capacity:
            penalty = 1000 * (total_weight - self.capacity)
            return -total_value + penalty
        return -total_value
    
    def random_solution(self) -> np.ndarray:
        solution = np.random.randint(0, 2, self.n_items)
        while np.sum(solution * self.weights) > self.capacity:
            selected_indices = np.where(solution == 1)[0]
            if len(selected_indices) == 0:
                break
            remove_idx = np.random.choice(selected_indices)
            solution[remove_idx] = 0
        return solution
    
    def is_valid(self, solution):
        # Kiểm tra là binary vector
        if not all(x in [0, 1] for x in solution):
            return False
        
        # Kiểm tra không vượt capacity
        total_weight = np.sum(solution * self.weights)
        return total_weight <= self.capacity
    
    def flip_bit(self, solution: np.ndarray, idx: int) -> np.ndarray:
        new_solution = solution.copy()
        new_solution[idx] = 1 - new_solution[idx]
        return new_solution
    
    def repair_solution(self, solution: np.ndarray) -> np.ndarray:
        """
        Hàm sửa nghiệm bằng cách loại bỏ các vật ít giá trị nhất/trọng lượng cho đến khi hợp lệ.
        """
        solution = solution.copy()
        while np.sum(solution * self.weights) > self.capacity:
            selected_indices = np.where(solution == 1)[0]
            if len(selected_indices) == 0:
                break
            ratios = self.values[selected_indices] / self.weights[selected_indices]
            remove_idx = selected_indices[np.argmin(ratios)]
            solution[remove_idx] = 0
        return solution
    
