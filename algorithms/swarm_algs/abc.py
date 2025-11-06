# Kiên
from algorithms.base_optimizer import SwarmOptimizer
import numpy as np
from config import ALGORITHM_PARAMS  # Import config để lấy tham số

class ArtificialBeeColony(SwarmOptimizer):
    """
    Triển khai thuật toán Tối ưu hóa Bầy ong Nhân tạo (Artificial Bee Colony).
    
    Kế thừa từ SwarmOptimizer.
    """
    
    def __init__(self, **kwargs):
        """
        Khởi tạo thuật toán ABC.
        Đọc các tham số (population_size, limit) từ file config.
        """
        # Lấy tham số riêng của ABC từ file config
        abc_params = ALGORITHM_PARAMS.get('abc', {})
        pop_size = abc_params.get('population_size', 30)  # SN = Số nguồn thức ăn
        self.limit = abc_params.get('limit', 10)         # Ngưỡng trial cho ong trinh sát
        
        # Gọi hàm __init__ của lớp cha (SwarmOptimizer)
        # pop_size ở đây được hiểu là số lượng nguồn thức ăn (SN),
        # cũng là số lượng ong Thợ và số lượng ong Quan sát.
        super().__init__(name="Artificial Bee Colony (ABC)", 
                         population_size=pop_size)
        
        # Bộ đếm trial cho từng nguồn thức ăn
        self.trial_counters = None
        self.dim = None

    def optimize(self, problem, max_iter: int = 100, **kwargs) -> tuple:
        """
        Hàm tối ưu hóa chính, được gọi bởi self.run().
        
        Args:
            problem: Đối tượng bài toán (phải là ContinuousProblem).
            max_iter: Số vòng lặp tối đa.
            
        Returns:
            Tuple (best_solution, best_fitness).
        """
        
        # 1. Khởi tạo
        self.dim = problem.dim
        # _initialize_population() đã có sẵn từ SwarmOptimizer
        # Nó sẽ tạo self.population, self.fitness_values,
        # cập nhật self.best_solution, self.best_fitness
        # và tăng self.function_evaluations
        self._initialize_population(problem)
        
        # Khởi tạo bộ đếm trial
        self.trial_counters = np.zeros(self.population_size)
        
        # Lưu lại fitness tốt nhất của thế hệ đầu tiên
        self._update_convergence()

        # 2. Vòng lặp chính
        for iter_num in range(max_iter):
            # === A. Giai đoạn Ong Thợ (Employed Bee Phase) ===
            for i in range(self.population_size):
                v_i = self._generate_candidate(i, problem)
                fit_v_i = problem.evaluate(v_i)
                self.function_evaluations += 1
                self._greedy_selection(i, v_i, fit_v_i)

            # === B. Giai đoạn Ong Quan Sát (Onlooker Bee Phase) ===
            probs = self._calculate_probabilities()
            for _ in range(self.population_size): # Số ong quan sát = SN
                i = self._select_source(probs) # Chọn nguồn thức ăn
                
                v_i = self._generate_candidate(i, problem)
                fit_v_i = problem.evaluate(v_i)
                self.function_evaluations += 1
                self._greedy_selection(i, v_i, fit_v_i)

            # === C. Giai đoạn Ong Trinh Sát (Scout Bee Phase) ===
            self._scout_phase(problem)
            
            # Cập nhật số vòng lặp và đường cong hội tụ
            self.iterations_done += 1
            self._update_convergence() # Ghi lại best_fitness của vòng này

        # 3. Trả về kết quả
        return self.best_solution, self.best_fitness

    # --- Các hàm private helper cho ABC ---

    def _generate_candidate(self, i: int, problem) -> np.ndarray:
        """Tạo giải pháp ứng viên v_i từ x_i."""
        # Chọn hàng xóm k ngẫu nhiên (k != i)
        k = np.random.randint(0, self.population_size)
        while k == i:
            k = np.random.randint(0, self.population_size)
            
        # Chọn chiều j ngẫu nhiên
        j = np.random.randint(0, self.dim)
        
        # Công thức tạo nghiệm mới
        phi = np.random.uniform(-1, 1)
        v_i = self.population[i].copy()
        v_i[j] = self.population[i, j] + phi * (self.population[i, j] - self.population[k, j])
        
        # Clip để đảm bảo v_i nằm trong biên
        return problem.clip_solution(v_i)

    def _greedy_selection(self, i: int, v_i: np.ndarray, fit_v_i: float):
        """Lựa chọn tham lam giữa v_i và x_i."""
        if fit_v_i < self.fitness_values[i]:
            # v_i tốt hơn, thay thế x_i
            self.population[i] = v_i
            self.fitness_values[i] = fit_v_i
            self.trial_counters[i] = 0
            
            # Cập nhật best global
            if fit_v_i < self.best_fitness:
                self.best_fitness = fit_v_i
                self.best_solution = v_i.copy()
        else:
            # v_i không tốt hơn, tăng bộ đếm trial
            self.trial_counters[i] += 1

    def _calculate_probabilities(self) -> np.ndarray:
        """Tính xác suất P_i cho ong quan sát (dùng fitness)."""
        # Vì là bài toán minimization, fitness càng nhỏ càng tốt
        # Chuyển đổi fitness (càng nhỏ) -> probability (càng lớn)
        max_fit = np.max(self.fitness_values)
        if max_fit == 0:
            # Xử lý trường hợp tất cả fitness = 0
            return np.ones(self.population_size) / self.population_size

        fitness_norm = 1.0 - (self.fitness_values / (max_fit + 1e-9))
        total_fit = np.sum(fitness_norm)
        
        if total_fit == 0:
            # Nếu tất cả fitness bằng nhau, P bằng nhau
            return np.ones(self.population_size) / self.population_size
            
        return fitness_norm / total_fit

    def _select_source(self, probs: np.ndarray) -> int:
        """Chọn nguồn thức ăn bằng Roulette Wheel Selection."""
        return np.random.choice(np.arange(self.population_size), p=probs)

    def _scout_phase(self, problem):
        """Giai đoạn ong trinh sát."""
        for i in range(self.population_size):
            if self.trial_counters[i] > self.limit:
                # Nguồn thức ăn i bị cạn kiệt, thay thế bằng nguồn ngẫu nhiên
                self.population[i] = problem.random_solution()
                self.fitness_values[i] = problem.evaluate(self.population[i])
                self.function_evaluations += 1
                self.trial_counters[i] = 0
                
                # Kiểm tra nếu nguồn mới là best
                if self.fitness_values[i] < self.best_fitness:
                    self.best_fitness = self.fitness_values[i]
                    self.best_solution = self.population[i].copy()