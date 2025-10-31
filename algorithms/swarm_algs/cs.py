import numpy as np
import math
from ..base_optimizer import SwarmOptimizer


class CuckooSearch(SwarmOptimizer):

    def __init__(self, population_size: int = 25, pa: float = 0.25, alpha: float = 0.01):
        """
        Args:
            population_size: số lượng tổ (nest)
            pa: xác suất loại bỏ tổ (discovery rate)
            alpha: hệ số bước (step size) cho Levy flight
        """
        super().__init__(name="Cuckoo Search", population_size=population_size)
        self.pa = pa
        self.alpha = alpha

    def _levy_flight(self, size, beta=1.5):
        """
        Sinh bước Levy flight cho tìm kiếm toàn cục.
        """
        sigma_u = (
            (math.gamma(1 + beta) * np.sin(np.pi * beta / 2))
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)

        u = np.random.normal(0, sigma_u, size)
        v = np.random.normal(0, 1, size)
        step = u / (np.abs(v) ** (1 / beta))
        return step

    def optimize(self, problem, max_iter=100, **kwargs):
        """
        Thực hiện tối ưu hóa bằng thuật toán Cuckoo Search.
        Args:
            problem: đối tượng bài toán có hàm evaluate() và random_solution()
            max_iter: số vòng lặp tối đa
        """
        # Khởi tạo quần thể ban đầu
        self._initialize_population(problem)
        n = self.population_size

        # Lặp tối ưu
        for t in range(max_iter):
            # Tạo tổ mới bằng Levy flight
            for i in range(n):
                current = self.population[i].copy()
                step = self._levy_flight(size=current.shape)  # sinh vector bước
                new_solution = current + self.alpha * step * (current - self.best_solution)

                # Giới hạn nghiệm trong phạm vi của problem (nếu có)
                new_solution = problem.clip_solution(new_solution)

                # Đánh giá
                new_fitness = problem.evaluate(new_solution)
                self.function_evaluations += 1

                # Thay thế nếu tốt hơn
                if new_fitness < self.fitness_values[i]:
                    self.population[i] = new_solution
                    self.fitness_values[i] = new_fitness

            # Loại bỏ tổ xấu theo xác suất pa
            abandon_mask = np.random.rand(n) < self.pa
            for i in range(n):
                if abandon_mask[i]:
                    new_nest = problem.random_solution()
                    new_fit = problem.evaluate(new_nest)
                    self.population[i] = new_nest
                    self.fitness_values[i] = new_fit
                    self.function_evaluations += 1

            # Cập nhật nghiệm tốt nhất
            best_idx = np.argmin(self.fitness_values)
            if self.fitness_values[best_idx] < self.best_fitness:
                self.best_fitness = self.fitness_values[best_idx]
                self.best_solution = self.population[best_idx].copy()

        return self.best_solution, self.best_fitness