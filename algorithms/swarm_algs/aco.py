from ..base_optimizer import SwarmOptimizer
import numpy as np
from typing import Any, Tuple
from problems.discrete_prob import TravelingSalesmanProblem as TSP


class AntColonyOptimization(SwarmOptimizer):
    """
    Ant Colony Optimization algorithm for solving TSP.
    """
    
    def __init__(self, name: str = "Ant Colony Optimization", 
                 population_size: int = 20,
                 alpha: float = 1.0, 
                 beta: float = 2.0, 
                 evaporation: float = 0.5,
                 pheromone_scale: float = 100):
        """
        Args:
            name: Tên thuật toán
            population_size: Số lượng kiến (n_ants)
            alpha: Hệ số quan trọng của pheromone (α)
            beta: Hệ số quan trọng của khoảng cách (β)
            evaporation: Tỷ lệ bay hơi pheromone (ρ)
            pheromone_scale: Hệ số Q trong công thức cập nhật pheromone
        """
        super().__init__(name, population_size)
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.Q = pheromone_scale
        self.pheromone = None
        
    def optimize(self, problem, max_iter: int = 100, **kwargs) -> Tuple[Any, float]:
        """
        Thực hiện tối ưu hóa bằng ACO.
        
        Args:
            problem: Bài toán TSP cần tối ưu
            max_iter: Số iteration tối đa
            **kwargs: Các tham số bổ sung
            
        Returns:
            Tuple[solution, fitness]: Tour tốt nhất và độ dài của nó
        """
        if not isinstance(problem, TSP):
            raise ValueError("ACO chỉ hoạt động với bài toán TSP")
            
        n_cities = problem.n_cities
        self.pheromone = np.ones((n_cities, n_cities))
        
        for iteration in range(max_iter):
            self.iterations_done += 1
            all_tours = []
            all_lengths = []
            
            # Mỗi con kiến xây dựng một tour
            for ant in range(self.population_size):
                tour = self._construct_tour(problem)
                length = problem.evaluate(tour)
                self.function_evaluations += 1
                
                all_tours.append(tour)
                all_lengths.append(length)
                
                # Cập nhật best nếu tìm được tour tốt hơn
                if length < self.best_fitness:
                    self.best_solution = tour.copy()
                    self.best_fitness = length
                    self._update_convergence()
            
            # Cập nhật pheromone
            self._update_pheromone(all_tours, all_lengths)
            
        return self.best_solution, self.best_fitness
    
    def _construct_tour(self, problem: TSP) -> np.ndarray:
        """
        Xây dựng một tour mới cho một con kiến.
        
        Args:
            problem: Bài toán TSP
            
        Returns:
            np.ndarray: Tour được xây dựng
        """
        n_cities = problem.n_cities
        tour = [np.random.randint(n_cities)]
        unvisited = set(range(n_cities)) - {tour[0]}
        
        while unvisited:
            current = tour[-1]
            probabilities = []
            
            # Tính xác suất cho các thành phố chưa thăm
            for city in unvisited:
                tau = self.pheromone[current, city] ** self.alpha
                eta = (1.0 / problem.distance_matrix[current, city]) ** self.beta
                probabilities.append(tau * eta)
            
            # Chuẩn hóa xác suất
            probabilities = np.array(probabilities)
            probabilities /= probabilities.sum()
            
            # Chọn thành phố tiếp theo
            next_city = np.random.choice(list(unvisited), p=probabilities)
            tour.append(next_city)
            unvisited.remove(next_city)
        
        return np.array(tour)
    
    def _update_pheromone(self, tours: list, lengths: list):
        """
        Cập nhật ma trận pheromone.
        
        Args:
            tours: Danh sách các tour
            lengths: Danh sách độ dài tương ứng của các tour
        """
        # Bay hơi pheromone
        self.pheromone *= (1 - self.evaporation)
        
        # Thêm pheromone mới
        for tour, length in zip(tours, lengths):
            for i in range(len(tour)):
                self.pheromone[tour[i], tour[(i+1)%len(tour)]] += self.Q / length
    
    def reset(self):
        """Reset trạng thái của thuật toán."""
        super().reset()
        self.pheromone = None