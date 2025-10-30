"""
Ví dụ: Implement PSO (Particle Swarm Optimization) sử dụng base class.
"""

import numpy as np
from ..base_optimizer import SwarmOptimizer


class PSO(SwarmOptimizer):
    """
    Particle Swarm Optimization - Thuật toán bầy đàn hạt.
    
    Tham khảo:
    Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
    """
    
    def __init__(self, population_size=30, w=0.7, c1=1.5, c2=1.5, v_max=None):
        """
        Khởi tạo PSO.
        
        Args:
            population_size: Số lượng particles
            w: Inertia weight (trọng số quán tính)
            c1: Cognitive coefficient (hệ số nhận thức cá nhân)
            c2: Social coefficient (hệ số xã hội)
            v_max: Vận tốc tối đa (None = tự động)
        """
        super().__init__(name="PSO", population_size=population_size)
        
        # PSO parameters
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max
        
        # PSO state variables
        self.velocities = None
        self.pbest = None  # Personal best positions
        self.pbest_fitness = None  # Personal best fitness values
        
    def optimize(self, problem, max_iter=100, **kwargs):
        """
        Chạy thuật toán PSO.
        
        Args:
            problem: Bài toán cần tối ưu
            max_iter: Số iteration tối đa
            
        Returns:
            Tuple[solution, fitness]: Nghiệm tốt nhất và fitness
        """
        # 1. Khởi tạo
        self._initialize(problem)
        
        # 2. Main loop
        for iteration in range(max_iter):
            # Update velocity và position
            self._update_velocity(problem)
            self._update_position(problem)
            
            # Evaluate fitness
            self._evaluate_population(problem)
            
            # Update personal best
            self._update_pbest()
            
            # Update global best
            self._update_gbest()
            
            # Lưu convergence
            self._update_convergence()
            self.iterations_done += 1
        
        return self.best_solution, self.best_fitness
    
    def _initialize(self, problem):
        """Khởi tạo particles và velocities."""
        # Khởi tạo population
        self._initialize_population(problem)
        
        # Get bounds
        lower, upper = problem.get_bounds()
        
        # Set v_max nếu chưa có
        if self.v_max is None:
            self.v_max = 0.2 * (upper - lower)
        
        # Khởi tạo velocities
        self.velocities = np.random.uniform(
            -self.v_max, self.v_max, 
            (self.population_size, problem.dim)
        )
        
        # Khởi tạo personal best
        self.pbest = self.population.copy()
        self.pbest_fitness = self.fitness_values.copy()
    
    def _update_velocity(self, problem):
        """Update velocities của tất cả particles."""
        r1 = np.random.rand(self.population_size, problem.dim)
        r2 = np.random.rand(self.population_size, problem.dim)
        
        # PSO velocity update formula
        cognitive = self.c1 * r1 * (self.pbest - self.population)
        social = self.c2 * r2 * (self.best_solution - self.population)
        
        self.velocities = (self.w * self.velocities + 
                          cognitive + social)
        
        # Clamp velocities
        self.velocities = np.clip(self.velocities, -self.v_max, self.v_max)
    
    def _update_position(self, problem):
        """Update positions của tất cả particles."""
        # Update positions
        self.population = self.population + self.velocities
        
        # Clip to bounds
        lower, upper = problem.get_bounds()
        self.population = np.clip(self.population, lower, upper)
    
    def _evaluate_population(self, problem):
        """Đánh giá fitness của toàn bộ population."""
        self.fitness_values = np.array([
            problem.evaluate(particle) 
            for particle in self.population
        ])
        self.function_evaluations += self.population_size
    
    def _update_pbest(self):
        """Update personal best cho mỗi particle."""
        # Tìm particles có fitness tốt hơn pbest
        improved = self.fitness_values < self.pbest_fitness
        
        # Update pbest cho những particles cải thiện
        self.pbest[improved] = self.population[improved].copy()
        self.pbest_fitness[improved] = self.fitness_values[improved]
    
    def _update_gbest(self):
        """Update global best."""
        min_idx = np.argmin(self.fitness_values)
        
        if self.fitness_values[min_idx] < self.best_fitness:
            self.best_fitness = self.fitness_values[min_idx]
            self.best_solution = self.population[min_idx].copy()
    
    def reset(self):
        """Reset PSO state."""
        super().reset()
        self.velocities = None
        self.pbest = None
        self.pbest_fitness = None


# Ví dụ sử dụng:
if __name__ == "__main__":
    from problems import SphereFunction
    
    # Tạo bài toán
    problem = SphereFunction(dim=10)
    
    # Tạo thuật toán
    pso = PSO(population_size=30, w=0.7, c1=1.5, c2=1.5)
    
    # Chạy tối ưu
    print(f"Running {pso.name} on {problem.name}...")
    result = pso.run(problem, max_iter=100)
    
    # In kết quả
    print(f"\nResults:")
    print(f"  Best fitness: {result['fitness']:.6f}")
    print(f"  Optimal value: {problem.optimal_value:.6f}")
    print(f"  Execution time: {result['execution_time']:.4f}s")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Function evaluations: {result['function_evaluations']}")
    
    # Lấy stats
    stats = pso.get_stats()
    print(f"\nStatistics:")
    print(f"  Convergence rate: {stats['convergence_rate']:.2%}")
    print(f"  Improvement: {stats['improvement']:.2f}%")
