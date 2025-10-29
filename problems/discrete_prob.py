import numpy as np
from typing import List, Tuple
from .base_problem import DiscreteProblem

class TravelingSalesmanProblem(DiscreteProblem):

    def __init__(self, n_cities=20, distance_matrix=None, seed=None):
        super().__init__(prob_name="TSP", n_cities=n_cities, distance_matrix=distance_matrix, seed=seed)
        self.n_cities = n_cities
        if distance_matrix is not None:
            self.distance_matrix = distance_matrix
        else:
            self.distance_matrix = self._generate_distance_matrix(seed)
        self.optimal_value = None

    def _generate_distance_matrix(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # Tạo tọa độ ngẫu nhiên cho các thành phố
        coords = np.random.rand(self.n_cities, 2) * 100
        # Tính khoảng cách Euclidean
        distances = np.zeros((self.n_cities, self.n_cities))
        for i in range(self.n_cities):
            for j in range(i + 1, self.n_cities):
                dist = np.linalg.norm(coords[i] - coords[j])
                distances[i, j] = dist
                distances[j, i] = dist
        return distances
    
    def evaluate(self, tour: np.ndarray) -> float:
        total_distance = 0
        for i in range(len(tour)):
            from_city = tour[i]
            to_city = tour[(i + 1) % len(tour)]
            total_distance += self.distance_matrix[from_city, to_city]
        return total_distance

    def random_solution(self) -> np.ndarray:
        return np.random.permutation(self.n_cities)
    
    def is_valid(self, tour: np.ndarray) -> bool:
        tour = np.array(tour)
        return (len(tour) == self.n_cities and
                len(set(tour)) == self.n_cities and
                all(0 <= city < self.n_cities for city in tour))
    
    def swap_cities(self, tour: np.ndarray, i: int, j: int) -> np.ndarray:
        new_tour = tour.copy()
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        return new_tour
    
    def two_opt(self, tour: np.ndarray, i: int, j: int) -> np.ndarray:
        new_tour = tour.copy()
        new_tour[i:j+1] = new_tour[i:j+1][::-1]
        return new_tour
    

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