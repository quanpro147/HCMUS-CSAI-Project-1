import numpy as np
import json
import os
from typing import List, Dict, Any
from testcases_loader import load_testcases
from utils import compute_basic_stats, compute_time_complexity, compute_robustness_metrics, compute_convergence_speed

from problems.discrete_prob import (
    TravelingSalesmanProblem,
    GridPathfindingProblem,
)

from algorithms import (
    AntColonyOptimization, 
    ACO_Pathfinder, 
    AStar, 
    SimulatedAnnealing,
    BFS
)

class DiscreteExperiment:
    """
    Class để chạy và quản lý experiments trên discrete problems.
    """
    
    def __init__(self, 
                 algorithms: List,
                 problems: List,
                 n_runs: int = 30,
                 max_iter: int = 100,
                 results_dir: str = "results"):
        """
        Args:
            algorithms: Danh sách các thuật toán cần test
            problems: Danh sách các bài toán
            n_runs: Số lần chạy mỗi thuật toán trên mỗi bài toán
            max_iter: Số iteration tối đa
            results_dir: Thư mục lưu kết quả
        """
        self.algorithms = algorithms
        self.problems = problems
        self.n_runs = n_runs
        self.max_iter = max_iter
        self.results_dir = results_dir
        
        # Tạo thư mục nếu chưa có
        os.makedirs(results_dir, exist_ok=True)
        
        # Lưu kết quả
        self.results = {}
        
    def run_single_experiment(self, algorithm, problem, run_id: int) -> Dict[str, Any]:
        """
        Chạy 1 lần experiment cho 1 thuật toán trên 1 bài toán.
        
        Returns:
            Dict chứa kết quả
        """
        algorithm.reset()
        result = algorithm.run(problem, max_iter=self.max_iter)
        return result
    
    def run_multiple_experiments(self, algorithm, problem) -> Dict[str, Any]:
        """
        Chạy 1 thuật toán nhiều lần trên 1 bài toán.
        
        Returns:
            Dict chứa tất cả kết quả và statistics
        """
        print(f"\nTesting {algorithm.name} on {problem.prob_name}...")
        
        all_results = []
        
        for i in range(self.n_runs):
            print(f"     Run {i+1}/{self.n_runs}...", end=" ")
            result_dict = self.run_single_experiment(algorithm, problem, i)
            print(f"Done! Fitness: {result_dict['fitness']:.2f} | Time: {result_dict['execution_time']:.4f}s")
            all_results.append(result_dict)
        
        # Tính statistics
        fitness_values = [r['fitness'] for r in all_results]
        time_values = [r['execution_time'] for r in all_results]
        
        # Get convergence curve and convert to list if needed
        best_convergence = all_results[np.argmin(fitness_values)]['convergence_curve']
        if isinstance(best_convergence, np.ndarray):
            best_convergence = best_convergence.tolist()
        
        # Calculate metrics
        fitness_stats = compute_basic_stats(fitness_values)
        time_stats = compute_time_complexity(time_values)
        robustness = compute_robustness_metrics(fitness_values)
        conv_speed = compute_convergence_speed(best_convergence)

        stats = {
            'n_runs': self.n_runs,
            'max_iter': self.max_iter,
            'fitness': fitness_stats,
            'time': time_stats,
            'robustness': robustness,
            'convergence_speed': conv_speed,
            'best_run_convergence': best_convergence
        }
        return stats
    
    def _is_algorithm_compatible(self, algorithm, problem) -> bool:
        """
        Kiểm tra xem thuật toán có tương thích với bài toán không.
        
        Args:
            algorithm: Instance của thuật toán
            problem: Instance của bài toán
            
        Returns:
            True nếu tương thích, False nếu không
        """
        # ACO và SimulatedAnnealing chỉ cho TSP
        if isinstance(problem, TravelingSalesmanProblem):
            return isinstance(algorithm, (AntColonyOptimization, SimulatedAnnealing, BFS))
        
        # ACO_Pathfinder và A* chỉ cho GridPathfinding
        elif isinstance(problem, GridPathfindingProblem):
            return isinstance(algorithm, (ACO_Pathfinder, AStar))
        
        return False
    
    def run(self):
        """
        Chạy tất cả các experiments.
        """
        print("RUNNING DISCRETE OPTIMIZATION EXPERIMENTS")
        print("="*80)
        
        for problem in self.problems:
            problem_key = getattr(problem, "test_id", problem.prob_name)
            print("\n" + "-"*80)
            print(f"Problem: {problem.prob_name} ({self._get_problem_size(problem)}) | ID: {problem_key}")
            print("-"*80)

            problem_results = {}

            for algorithm in self.algorithms:
                # Kiểm tra tương thích
                if not self._is_algorithm_compatible(algorithm, problem):
                    continue
                
                stats = self.run_multiple_experiments(algorithm, problem)
                problem_results[algorithm.name] = stats
            
            self.results[problem_key] = problem_results

        self.summary()
        self.save_results()
        
    def save_results(self, filename: str = None):
        """
        Lưu kết quả ra file JSON.
        """
        if filename is None:
            filename = f"discrete_results.json"
        
        filepath = os.path.join(self.results_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.results, f, indent=4)
            print(f"\nResults saved to: {filepath}")
        except Exception as e:
            print(f"\nError saving results to {filepath}: {e}")
    
    def summary(self):
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        
        for problem_id, problem_results in self.results.items():
            print(f"\n\nProblem: {problem_id}")
            print(f"{'Algorithm':<25} {'Mean Fitness':<15} {'Std Fitness':<12} {'Best Fitness':<12} {'Mean Time (s)':<10}")
            print("-"*80)
            
            sorted_results = sorted(
                problem_results.items(),
                key=lambda x: (x[1]['fitness']['mean'], x[1]['time']['mean'])
            )
            
            for algo_name, stats in sorted_results:
                print(f"{algo_name:<25} "
                    f"{stats['fitness']['mean']:<15.6f} "
                    f"{stats['fitness']['std']:<12.6f} "
                    f"{stats['fitness']['min']:<12.6f} "
                    f"{stats['time']['mean']:<10.4f}")
            
            if sorted_results:
                best_algo = sorted_results[0][0]
                best_fitness = sorted_results[0][1]['fitness']['mean']
                best_time = sorted_results[0][1]['time']['mean']
                print(f"\n  Best: {best_algo} (fitness={best_fitness:.6f}, time={best_time:.4f}s)")

    
    def _get_problem_size(self, problem):
        """
        Helper lấy kích thước bài toán để in ra.
        """
        if isinstance(problem, TravelingSalesmanProblem):
            return f"{problem.n_cities} cities"
        elif isinstance(problem, GridPathfindingProblem):
            return f"{problem.height}x{problem.width} grid"
        return "N/A"


def main():

    # Load testcase
    dis_problems = load_testcases("testcases/discrete_testcases.json")
    
    dis_algorithms = [
        AntColonyOptimization(),
        SimulatedAnnealing(),
        ACO_Pathfinder(),
        AStar(),
        BFS()
    ]
    disc_experiment = DiscreteExperiment(
        algorithms=dis_algorithms,
        problems=dis_problems,
        n_runs=10, 
        max_iter=100
    )
    disc_experiment.run()


if __name__ == "__main__":
    main()