import numpy as np
import json
import time
import os
from datetime import datetime
from typing import List, Dict, Any
from config import EXPERIMENT_CONFIG, ALGORITHM_PARAMS

from problems.discrete_prob import (
    TravelingSalesmanProblem,
    GridPathfindingProblem,
)

from algorithms import AntColonyOptimization, ACO_Pathfinder, AStar
from utils.metrics import compute_basic_stats, compute_time_complexity, compute_robustness_metrics, compute_convergence_speed
from utils.data_loader import load_tsp_case, load_grid_case


class DiscreteExperiment:
    """
    Class để chạy và quản lý các thí nghiệm trên bài toán Rời Rạc.
    """
    
    def __init__(self, n_runs: int, max_iter: int, results_dir: str, use_testcase: bool = True):
        self.n_runs = n_runs
        self.max_iter = max_iter
        self.results_dir = results_dir
        self.use_testcase = use_testcase
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
        tsp_sizes = EXPERIMENT_CONFIG.get('tsp_sizes', [10, 20])
        for n_cities in tsp_sizes:
            if self.use_testcase:
                # Load test case từ file
                tsp_data = load_tsp_case('testcases/tsp_test.json')
                
                # Tạo TSP - class TravelingSalesmanProblem tự xử lý coords/distance_matrix
                tsp_problem = TravelingSalesmanProblem(
                    n_cities=tsp_data.get('n_cities', n_cities),
                    coords=tsp_data.get('coords'),
                    distance_matrix=tsp_data.get('distance_matrix')
                )
                
                self.experiment_setup.append({
                    'problem': tsp_problem,
                    'algorithms': [AntColonyOptimization],
                    'max_iter': self.max_iter
                })
            else:
                # Tạo TSP ngẫu nhiên
                self.experiment_setup.append({
                    'problem': TravelingSalesmanProblem(n_cities=n_cities),
                    'algorithms': [AntColonyOptimization],
                    'max_iter': self.max_iter
                })

        # === Thí nghiệm 2: So sánh trên Grid Pathfinding ===
        grid_sizes = EXPERIMENT_CONFIG.get('grid_sizes', [(10, 10)])
        for (h, w) in grid_sizes:
            if self.use_testcase:
                # Load grid từ file
                grid_data = load_grid_case('testcases/grid_test.json')
                
                # Tạo GridPathfindingProblem từ data
                grid_problem = GridPathfindingProblem(
                    grid=grid_data.get('grid'),
                    start=tuple(grid_data.get('start', [0, 0])),
                    goal=tuple(grid_data.get('goal', [h-1, w-1]))
                )
            else:
                # Tạo grid ngẫu nhiên
                grid_problem = GridPathfindingProblem(
                    height=h,
                    width=w,
                    obstacle_ratio=0.2,
                    seed=None
                )
            
            self.experiment_setup.append({
                'problem': grid_problem,
                'algorithms': [ACO_Pathfinder, AStar],
                'max_iter': EXPERIMENT_CONFIG.get('max_iter', 100)
            })

    def run(self):
        """
        Chạy tất cả các thí nghiệm đã thiết lập.
        """
        print("RUNNING DISCRETE OPTIMIZATION EXPERIMENTS")
        print("="*80)
        
        total_runs = 0
        
        for exp in self.experiment_setup:
            problem = exp['problem']
            algorithms = exp['algorithms']
            max_iter = exp['max_iter']

            print(f"\nProblem: {problem.prob_name} (Size: {self._get_problem_size(problem)})")
            
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
                print(f"Running {algo_instance.name} on {problem.prob_name}...")

                current_max_iter = max_iter 
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
        self.summary()
        self._save_results()

    def summary(self):
        print("DISCRETE EXPERIMENT SUMMARY")
        print(f"{'Problem':<20} {'Algorithm':<25} {'Mean Fitness':<15} {'Std Fitness':<12} {'Best Fitness':<12} {'Mean Time (s)':<14}")
        print("-"*100)
        for result in self.results:
            print(f"{result['problem']:<20} {result['algorithm']:<25} "
                  f"{result['fitness']['mean']:<15.6f} "
                  f"{result['fitness']['std']:<12.6f} "
                  f"{result['fitness']['min']:<12.6f} "
                  f"{result['time']['mean']:<14.4f}")

        # Highlight best per problem
        problems = set(r['problem'] for r in self.results)
        for p in problems:
            best = min((r for r in self.results if r['problem']==p), key=lambda r: r['fitness']['mean'])
            print(f"\n  🏆 Best for {p}: {best['algorithm']} (fitness={best['fitness']['mean']:.6f})")

    def _get_problem_size(self, problem):
        """Helper lấy kích thước bài toán để in ra."""
        if isinstance(problem, TravelingSalesmanProblem):
            return f"{problem.n_cities} cities"
        if isinstance(problem, GridPathfindingProblem):
            return f"{problem.height}x{problem.width} grid"
        return "N/A"

    def _save_summary(self, problem, algo, run_results):
        fitness_arr = np.array(run_results['fitness_list'])
        time_arr = np.array(run_results['time_list'])
        evals_arr = np.array(run_results['evals_list'])
        conv_curves = run_results['convergence_curves']

        fitness_stats = compute_basic_stats(fitness_arr)
        time_stats = compute_time_complexity(time_arr)
        robustness = compute_robustness_metrics(fitness_arr)
        mean_convergence = np.mean(conv_curves, axis=0).tolist()
        mean_conv_speed = compute_convergence_speed(mean_convergence)

        summary = {
            'problem': problem.prob_name,
            'problem_size': self._get_problem_size(problem),
            'algorithm': algo.name,
            'n_runs': self.n_runs,
            'fitness': fitness_stats,
            'time': time_stats,
            'robustness': robustness,
            'convergence_speed': mean_conv_speed,
            'mean_convergence_curve': mean_convergence
        }
        self.results.append(summary)

    def _save_results(self, filename: str = None):

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"discrete/results_{timestamp}.json"
        
        filepath = os.path.join(self.results_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.results, f, indent=4)
            print(f"\nResults saved to: {filepath}")
        except Exception as e:
            print(f"\nError saving results to {filepath}: {e}")


def main():
    
    experiment = DiscreteExperiment(
        n_runs=10,
        max_iter=100,
        results_dir='results'
    )
    experiment.run()
    experiment.summary()

if __name__ == "__main__":
    main()