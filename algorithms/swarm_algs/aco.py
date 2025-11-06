
import numpy as np
from typing import Any, Tuple, List

# Import lớp base và config
from algorithms.base_optimizer import SwarmOptimizer
from config import ALGORITHM_PARAMS

# Import các Problem mà 2 lớp ACO này sẽ giải
from problems.discrete_prob import TravelingSalesmanProblem, GridPathfindingProblem

# ==============================================================================
# ===   LỚP 1: ACO CHO BÀI TOÁN TSP (Đã sửa lỗi)   ===
# ==============================================================================

class AntColonyOptimization(SwarmOptimizer):
    """
    Ant Colony Optimization algorithm for solving TSP (Đồ thị đầy đủ).
    """
    
    def __init__(self, **kwargs):
        """
        Khởi tạo bằng cách đọc thông số từ config.py
        """
        # Lấy tham số riêng của ACO từ config
        aco_params = ALGORITHM_PARAMS.get('aco', {})
        pop_size = aco_params.get('n_ants', 20)
        self.alpha = aco_params.get('alpha', 1.0)
        self.beta = aco_params.get('beta', 2.0)
        self.evaporation = aco_params.get('rho', 0.5) # 'rho' trong config
        self.Q = aco_params.get('pheromone_scale', 100)

        super().__init__(name="ACO (for TSP)", population_size=pop_size)
        self.pheromone = None
        
    def optimize(self, problem: TravelingSalesmanProblem, max_iter: int = 100, **kwargs) -> Tuple[Any, float]:
        """
        Thực hiện tối ưu hóa bằng ACO.
        """
        if not isinstance(problem, TravelingSalesmanProblem):
            raise ValueError("AntColonyOptimization chỉ hoạt động với TravelingSalesmanProblem")
            
        n_cities = problem.n_cities
        self.pheromone = np.ones((n_cities, n_cities))
        
        for iteration in range(max_iter):
            self.iterations_done += 1
            all_tours = []
            all_lengths = []
            
            for ant in range(self.population_size):
                tour = self._construct_tour(problem)
                length = problem.evaluate(tour)
                self.function_evaluations += 1
                
                all_tours.append(tour)
                all_lengths.append(length)
                
                if length < self.best_fitness:
                    self.best_solution = tour.copy()
                    self.best_fitness = length
            
            self._update_pheromone(all_tours, all_lengths)
            self._update_convergence() # Cập nhật best fitness của vòng này
            
        return self.best_solution, self.best_fitness
    
    def _construct_tour(self, problem: TravelingSalesmanProblem) -> np.ndarray:
        """
        Xây dựng một tour mới cho một con kiến (logic TSP).
        """
        n_cities = problem.n_cities
        tour = [np.random.randint(n_cities)]
        unvisited = set(range(n_cities)) - {tour[0]}
        
        while unvisited:
            current = tour[-1]
            probabilities = []
            cities_list = list(unvisited)
            
            for city in cities_list:
                tau = self.pheromone[current, city] ** self.alpha
                # Thêm 1e-9 để tránh chia cho 0
                eta = (1.0 / (problem.distance_matrix[current, city] + 1e-9)) ** self.beta
                probabilities.append(tau * eta)
            
            probabilities = np.array(probabilities)
            prob_sum = probabilities.sum()
            if prob_sum == 0:
                probabilities = np.ones_like(probabilities) / len(probabilities)
            else:
                probabilities /= prob_sum
            
            next_city = np.random.choice(cities_list, p=probabilities)
            tour.append(next_city)
            unvisited.remove(next_city)
        
        return np.array(tour)
    
    def _update_pheromone(self, tours: list, lengths: list):
        """
        Cập nhật ma trận pheromone (logic TSP).
        """
        self.pheromone *= (1 - self.evaporation)
        
        for tour, length in zip(tours, lengths):
            if length == 0: continue
            for i in range(len(tour)):
                city1 = tour[i]
                city2 = tour[(i+1)%len(tour)]
                self.pheromone[city1, city2] += self.Q / length
                self.pheromone[city2, city1] += self.Q / length
    
    def reset(self):
        super().reset()
        self.pheromone = None

# ==============================================================================
# ===   LỚP 2: ACO CHO BÀI TOÁN TÌM ĐƯỜNG (Để so sánh với A*)   ===
# ==============================================================================

class ACO_Pathfinder(SwarmOptimizer):
    """
    Ant Colony Optimization algorithm for solving Pathfinding (Grid).
    Pheromone được lưu trên các NÚT (ô) của lưới.
    """
    
    def __init__(self, **kwargs):
        """
        Khởi tạo bằng cách đọc thông số từ config.py
        """
        # Dùng chung tham số với ACO chuẩn
        aco_params = ALGORITHM_PARAMS.get('aco_pathfinder', ALGORITHM_PARAMS.get('aco'))
        pop_size = aco_params.get('n_ants', 30)
        self.alpha = aco_params.get('alpha', 1.0)
        self.beta = aco_params.get('beta', 2.0)
        self.evaporation = aco_params.get('rho', 0.5)
        self.Q = aco_params.get('pheromone_scale', 100)

        super().__init__(name="ACO (for Pathfinding)", population_size=pop_size)
        self.pheromone = None
        self.start_node = None
        self.goal_node = None

    def optimize(self, problem: GridPathfindingProblem, max_iter: int = 100, **kwargs) -> Tuple[Any, float]:
        """
        Thực hiện tìm đường bằng ACO.
        """
        if not isinstance(problem, GridPathfindingProblem):
            raise ValueError("ACO_Pathfinder chỉ hoạt động với GridPathfindingProblem")
            
        self.pheromone = np.ones((problem.height, problem.width))
        self.start_node = problem.get_start_state()
        self.goal_node = problem.goal_pos        
        for iteration in range(max_iter):
            self.iterations_done += 1
            all_paths = []
            all_lengths = []
            
            for ant in range(self.population_size):
                path = self._construct_path(problem)
                self.function_evaluations += 1
                
                if path is not None:
                    length = len(path) - 1 # Cost = số bước đi
                    all_paths.append(path)
                    all_lengths.append(length)
                
                    if length < self.best_fitness:
                        self.best_solution = path
                        self.best_fitness = length
            
            self._update_pheromone(all_paths, all_lengths)
            self._update_convergence()
            
        return self.best_solution, self.best_fitness

    def _construct_path(self, problem: GridPathfindingProblem) -> List[tuple]:
        """
        Xây dựng một đường đi cho một con kiến (logic Pathfinding).
        """
        path = [self.start_node]
        visited = {self.start_node}
        current = self.start_node
        max_steps = problem.height * problem.width 
        
        while current != self.goal_node and len(path) < max_steps:
            valid_neighbors = []
            for state, cost in problem.get_neighbors(current):
                if state not in visited:
                    valid_neighbors.append(state)
            
            if not valid_neighbors:
                return None 

            probabilities = []
            for state in valid_neighbors:
                y, x = state
                tau = self.pheromone[y, x] ** self.alpha
                h = problem.get_heuristic(state) 
                eta = (1.0 / (h + 1e-9)) ** self.beta
                probabilities.append(tau * eta)
            
            probabilities = np.array(probabilities)
            prob_sum = probabilities.sum()
            if prob_sum == 0:
                probabilities = np.ones_like(probabilities) / len(probabilities)
            else:
                probabilities /= prob_sum
            
            next_node = valid_neighbors[np.random.choice(len(valid_neighbors), p=probabilities)]
            path.append(next_node)
            visited.add(next_node)
            current = next_node
        
        return path if current == self.goal_node else None

    def _update_pheromone(self, paths: list, lengths: list):
        """
        Cập nhật ma trận pheromone (logic Pathfinding).
        """
        self.pheromone *= (1 - self.evaporation)
        
        for path, length in zip(paths, lengths):
            if length == 0: continue
            deposit_amount = self.Q / length
            for (y, x) in path:
                self.pheromone[y, x] += deposit_amount
    
    def reset(self):
        super().reset()
        self.pheromone = None
        self.start_node = None
        self.goal_node = None