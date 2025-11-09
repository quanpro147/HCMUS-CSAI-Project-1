from ..base_optimizer import SwarmOptimizer
import numpy as np
from typing import Any, Tuple
from config import ALGORITHM_PARAMS


class ParticleSwarmOptimization(SwarmOptimizer):
    """
    Particle Swarm Optimization (PSO) algorithm.
    """

    def __init__(self, 
                 name: str = "Particle Swarm Optimization",
                 population_size=None,
                 w=None,          # Hệ số quán tính
                 c1=None,         # Hệ số học hỏi cá nhân
                 c2=None):        # Hệ số học hỏi xã hội
        """
        Args:
            name: Tên thuật toán
            population_size: Số lượng hạt
            w: Hệ số quán tính (inertia weight)
            c1: Hệ số học hỏi cá nhân
            c2: Hệ số học hỏi xã hội
        """
        pso_params = ALGORITHM_PARAMS.get('pso', {})
        population_size = population_size if population_size is not None else pso_params.get('population_size', 30)
        w = w if w is not None else pso_params.get('w', 0.7)
        c1 = c1 if c1 is not None else pso_params.get('c1', 1.5)
        c2 = c2 if c2 is not None else pso_params.get('c2', 1.5)
        
        super().__init__(name, population_size)
        self.w = w
        self.c1 = c1
        self.c2 = c2

        # Các biến của PSO
        self.positions = None
        self.velocities = None
        self.pbest = None
        self.pbest_fitness = None
        self.gbest = None
        self.gbest_fitness = np.inf

    def optimize(self, problem, max_iter: int = 100, **kwargs) -> Tuple[Any, float]:
        """
        Thực hiện tối ưu hóa bằng PSO.

        Args:
            problem: Bài toán cần tối ưu hóa (có các thuộc tính: lower_bound, upper_bound, evaluate())
            max_iter: Số vòng lặp tối đa

        Returns:
            Tuple[solution, fitness]: Nghiệm tối ưu và giá trị hàm mục tiêu
        """

        dim = problem.dim
        lb, ub = np.array(problem.lower_bound), np.array(problem.upper_bound)

        # Khởi tạo quần thể
        self.positions = np.random.uniform(lb, ub, (self.population_size, dim))
        self.velocities = np.zeros((self.population_size, dim))

        # Khởi tạo pbest, gbest
        fitness = np.array([problem.evaluate(x) for x in self.positions])
        self.function_evaluations += self.population_size

        self.pbest = self.positions.copy()
        self.pbest_fitness = fitness.copy()

        best_idx = np.argmin(fitness)
        self.gbest = self.positions[best_idx].copy()
        self.gbest_fitness = fitness[best_idx]
        self.best_solution = self.gbest.copy()
        self.best_fitness = self.gbest_fitness

        # --- Quá trình tối ưu ---
        for iteration in range(max_iter):
            self.iterations_done += 1

            # Tính vận tốc và vị trí mới
            r1, r2 = np.random.rand(self.population_size, dim), np.random.rand(self.population_size, dim)
            self.velocities = (
                self.w * self.velocities
                + self.c1 * r1 * (self.pbest - self.positions)
                + self.c2 * r2 * (self.gbest - self.positions)
            )

            # Cập nhật vị trí
            self.positions += self.velocities

            # Giữ vị trí trong giới hạn
            self.positions = np.clip(self.positions, lb, ub)

            # Đánh giá
            fitness = np.array([problem.evaluate(x) for x in self.positions])
            self.function_evaluations += self.population_size

            # Cập nhật pbest
            better_mask = fitness < self.pbest_fitness
            self.pbest[better_mask] = self.positions[better_mask]
            self.pbest_fitness[better_mask] = fitness[better_mask]

            # Cập nhật gbest
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.gbest_fitness:
                self.gbest = self.positions[best_idx].copy()
                self.gbest_fitness = fitness[best_idx]

                self.best_solution = self.gbest.copy()
                self.best_fitness = self.gbest_fitness
                self._update_convergence()

        return self.best_solution, self.best_fitness

    def reset(self):
        """Reset trạng thái của PSO."""
        super().reset()
        self.positions = None
        self.velocities = None
        self.pbest = None
        self.pbest_fitness = None
        self.gbest = None
        self.gbest_fitness = np.inf
