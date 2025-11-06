
import numpy as np
import json
import time
import os
from datetime import datetime
from typing import List, Dict, Any

# Import config
from config import EXPERIMENT_CONFIG, ALGORITHM_PARAMS

# Import problems
from problems.discrete_prob import (
    TravelingSalesmanProblem,
    GridPathfindingProblem,
)

# Import algorithms
from algorithms.swarm_algs.aco import AntColonyOptimization, ACO_Pathfinder
from algorithms.traditional_algs.a_star import AStar
# (Bạn có thể import thêm Hill Climbing, SA... ở đây nếu muốn so sánh trên TSP)


class DiscreteExperiment:
    """
    Class để chạy và quản lý các thí nghiệm trên bài toán Rời Rạc.
    """
    
    def __init__(self, n_runs: int, results_dir: str):
        self.n_runs = n_runs
        self.results_dir = results_dir
        self.results = []
        
        # Tạo thư mục results nếu chưa có
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Định nghĩa các cặp (Thuật toán, Bài toán) và kích thước
        self.experiment_setup = []
        self._setup_experiments()

    def _setup_experiments(self):
        """
        Định nghĩa các thí nghiệm cần chạy (Task 2.2).
        Đây là nơi bạn ghép (Thuật toán, Bài toán, Cỡ)
        """
        
        # === Thí nghiệm 1: So sánh trên TSP ===
        # (Chỉ có ACO, bạn có thể thêm SA, GA... vào đây)
        tsp_sizes = EXPERIMENT_CONFIG.get('tsp_sizes', [10, 20])
        for n_cities in tsp_sizes:
            self.experiment_setup.append({
                'problem': TravelingSalesmanProblem(n_cities=n_cities),
                'algorithms': [AntColonyOptimization],
                'max_iter': EXPERIMENT_CONFIG.get('max_iter', 100)
            })

        # === Thí nghiệm 2: So sánh trên Grid Pathfinding ===
        # (ACO_Pathfinder vs A*)
        grid_sizes = EXPERIMENT_CONFIG.get('grid_sizes', [(10, 10)])
        for (h, w) in grid_sizes:
            # Tạo 1 grid đơn giản với start (0,0) và goal (h-1, w-1)
            grid = np.zeros((h, w))
            start = (0, 0)
            goal = (h-1, w-1)
            # (Bạn có thể thêm tường (obstacles) vào grid ở đây nếu muốn)
            
            self.experiment_setup.append({
                'problem': GridPathfindingProblem(grid=grid, start=start, goal=goal),
                'algorithms': [ACO_Pathfinder, AStar],
                'max_iter': EXPERIMENT_CONFIG.get('max_iter', 100) # Dùng max_iter chuẩn
            })

    def run(self):
        """
        Chạy tất cả các thí nghiệm đã thiết lập.
        """
        print("\n" + "="*80)
        print("    🔬 BẮT ĐẦU CHẠY DISCRETE EXPERIMENTS")
        print("="*80)
        
        total_runs = 0
        
        for exp in self.experiment_setup:
            problem = exp['problem']
            algorithms = exp['algorithms']
            max_iter = exp['max_iter']

            print(f"\n--- 📊 Problem: {problem.prob_name} (Size: {self._get_problem_size(problem)}) ---")
            
            for AlgoClass in algorithms:
                # Lấy params từ config
                algo_params = {}
                if AlgoClass == AntColonyOptimization:
                    algo_params = ALGORITHM_PARAMS.get('aco', {})
                elif AlgoClass == ACO_Pathfinder:
                    algo_params = ALGORITHM_PARAMS.get('aco_pathfinder', {})
                elif AlgoClass == AStar:
                    algo_params = ALGORITHM_PARAMS.get('a_star', {})
                
                algo_instance = AlgoClass(**algo_params)
                print(f"  -> 🏃 Running Algorithm: {algo_instance.name}")

                # Gán max_iter chuẩn
                current_max_iter = max_iter 

                # KIỂM TRA ĐẶC BIỆT: A* dùng max_iter làm "giới hạn duyệt nút"
                if AlgoClass == AStar:
                    current_max_iter = ALGORITHM_PARAMS.get('a_star', {}).get('max_iter', 50000)
                    print(f"     (Using special max_iter for A*: {current_max_iter})")
                
                # Nơi lưu kết quả của n_runs (Task 1.3: Robustness)
                run_results = {
                    'fitness_list': [],
                    'time_list': [],
                    'evals_list': [],
                    'convergence_curves': [],
                }
                
                for i in range(self.n_runs):
                    print(f"     Run {i+1}/{self.n_runs}...", end=" ")
                    
                    # Chạy thuật toán
                    result_dict = algo_instance.run(problem=problem, max_iter=current_max_iter)
                    
                    # Thu thập metrics
                    run_results['fitness_list'].append(result_dict['fitness'])
                    run_results['time_list'].append(result_dict['execution_time'])
                    run_results['evals_list'].append(result_dict['function_evaluations'])
                    run_results['convergence_curves'].append(result_dict['convergence_curve'])
                    
                    total_runs += 1
                    print(f"Done! Fitness: {result_dict['fitness']:.2f} | Time: {result_dict['execution_time']:.4f}s")
                
                # Tính toán Robustness
                self._save_summary(problem, algo_instance, run_results)

        print("\n✅ DISCRETE EXPERIMENTS COMPLETED!")
        print(f"   Tổng số lần chạy: {total_runs}")
        self._save_to_json()

    def _get_problem_size(self, problem):
        """Helper lấy kích thước bài toán để in ra."""
        if isinstance(problem, TravelingSalesmanProblem):
            return f"{problem.n_cities} cities"
        if isinstance(problem, GridPathfindingProblem):
            return f"{problem.height}x{problem.width} grid"
        return "N/A"

    def _save_summary(self, problem, algo, run_results):
        """
        Tính toán Mean, Std và lưu vào self.results
        Đây là phần thực thi Task 1.
        """
        fitness_arr = np.array(run_results['fitness_list'])
        time_arr = np.array(run_results['time_list'])
        evals_arr = np.array(run_results['evals_list'])

        summary = {
            'problem': problem.prob_name,
            'problem_size': self._get_problem_size(problem),
            'algorithm': algo.name,
            'n_runs': self.n_runs,

            # Metric: Robustness (Mean và Std)
            'fitness_mean': float(np.mean(fitness_arr)),
            'fitness_std': float(np.std(fitness_arr)),
            'fitness_best': float(np.min(fitness_arr)),
            'fitness_worst': float(np.max(fitness_arr)),

            # Metric: Computational Time (Mean và Std)
            'time_mean': float(np.mean(time_arr)),
            'time_std': float(np.std(time_arr)),

            # Evals có thể là số nguyên
            'evals_mean': float(np.mean(evals_arr)),
            'evals_std': float(np.std(evals_arr)),

            # Metric: Convergence (lấy đường cong hội tụ trung bình)
            # .tolist() đã tự động chuyển đổi kiểu dữ liệu
            'convergence_mean': np.mean(run_results['convergence_curves'], axis=0).tolist(),
        }

        self.results.append(summary)

    def _save_to_json(self):
        """Lưu file JSON kết quả."""
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.results_dir, f"discrete_results_{now}.json")
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=4)
            print(f"\n💾 Kết quả đã lưu vào: {filename}")
        except Exception as e:
            print(f"\n❌ Lỗi khi lưu file JSON: {e}")
