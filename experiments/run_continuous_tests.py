"""
Run experiments on continuous optimization problems.
So sánh các thuật toán Swarm Intelligence vs Traditional trên bài toán liên tục.
"""

import numpy as np
import json
import time
from datetime import datetime
from typing import List, Dict, Any
import os

# Import problems
from problems import (
    SphereFunction, 
    RastriginFunction, 
    RosenbrockFunction,
    AckleyFunction
)

# Import algorithms
from algorithms import FireflyAlgorithm
# from algorithms import PSO
# from algorithms import ABC
from algorithms import HillClimbing


class ContinuousExperiment:
    """
    Class để chạy và quản lý experiments trên continuous problems.
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
        print(f"    Run {run_id + 1}/{self.n_runs}...", end='')
        
        # Reset thuật toán
        algorithm.reset()
        
        # Chạy
        result = algorithm.run(problem, max_iter=self.max_iter)
        
        print(f" fitness={result['fitness']:.6f}, time={result['execution_time']:.4f}s")
        
        return result
    
    def run_algorithm_on_problem(self, algorithm, problem) -> Dict[str, Any]:
        """
        Chạy 1 thuật toán nhiều lần trên 1 bài toán.
        
        Returns:
            Dict chứa tất cả kết quả và statistics
        """
        print(f"\nTesting {algorithm.name} on {problem.prob_name}...")
        
        all_results = []
        
        for run_id in range(self.n_runs):
            result = self.run_single_experiment(algorithm, problem, run_id)
            all_results.append(result)
        
        # Tính statistics
        fitness_values = [r['fitness'] for r in all_results]
        time_values = [r['execution_time'] for r in all_results]
        iterations_values = [r['iterations'] for r in all_results]
        evaluations_values = [r['function_evaluations'] for r in all_results]
        
        stats = {
            'algorithm': algorithm.name,
            'problem': problem.prob_name,
            'n_runs': self.n_runs,
            'max_iter': self.max_iter,
            
            # Fitness statistics
            'fitness': {
                'mean': float(np.mean(fitness_values)),
                'std': float(np.std(fitness_values)),
                'min': float(np.min(fitness_values)),
                'max': float(np.max(fitness_values)),
                'median': float(np.median(fitness_values)),
                'q25': float(np.percentile(fitness_values, 25)),
                'q75': float(np.percentile(fitness_values, 75)),
                'all_values': fitness_values
            },
            
            # Time statistics
            'time': {
                'mean': float(np.mean(time_values)),
                'std': float(np.std(time_values)),
                'min': float(np.min(time_values)),
                'max': float(np.max(time_values)),
                'all_values': time_values
            },
            
            # Other metrics
            'iterations': {
                'mean': float(np.mean(iterations_values))
            },
            'evaluations': {
                'mean': float(np.mean(evaluations_values))
            },
            
            # Convergence curves (lấy từ run tốt nhất)
            'best_run_convergence': all_results[np.argmin(fitness_values)]['convergence_curve'],
            
            # Error from optimal
            'error_from_optimal': {
                'mean': float(np.mean(fitness_values) - problem.optimal_value),
                'min': float(np.min(fitness_values) - problem.optimal_value)
            }
        }
        
        return stats
    
    def run_all_experiments(self):
        """
        Chạy tất cả experiments: tất cả thuật toán trên tất cả bài toán.
        """
        print("RUNNING CONTINUOUS OPTIMIZATION EXPERIMENTS")
        print(f"Number of algorithms: {len(self.algorithms)}")
        print(f"Number of problems: {len(self.problems)}")
        print(f"Runs per experiment: {self.n_runs}")
        print(f"Max iterations: {self.max_iter}")
        
        start_time = time.time()
        
        # Chạy từng problem
        for problem in self.problems:
            print(f"\nProblem: {problem.prob_name} (dim={problem.dim})")

            problem_results = {}
            
            # Chạy từng thuật toán
            for algorithm in self.algorithms:
                stats = self.run_algorithm_on_problem(algorithm, problem)
                problem_results[algorithm.name] = stats
            
            # Lưu kết quả
            self.results[problem.prob_name] = problem_results
        
        total_time = time.time() - start_time

        print(f"\nALL EXPERIMENTS COMPLETED!")
        print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    def run(self):
        """
        Method chính để chạy toàn bộ workflow: run -> summary -> save.
        Chỉ cần gọi method này từ main.py
        """
        # 1. Chạy experiments
        self.run_all_experiments()
        
        # 2. In summary
        self.print_summary()
        
        # 3. Lưu kết quả
        filepath = self.save_results()
        
        return filepath
        
    def save_results(self, filename: str = None):
        """
        Lưu kết quả ra file JSON.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{timestamp}.json"
        
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=4)
        
        print(f"\nResults saved to: {filepath}")
        
        return filepath
    
    def print_summary(self):
        """
        In tóm tắt kết quả.
        """
        print("EXPERIMENT SUMMARY")
        
        for problem_name, problem_results in self.results.items():
            print(f"\n {problem_name}:")
            print(f"{'Algorithm':<25} {'Mean Fitness':<15} {'Std':<12} {'Best':<12} {'Time (s)':<10}")
            print("-"*80)
            
            # Sort by mean fitness
            sorted_results = sorted(
                problem_results.items(), 
                key=lambda x: x[1]['fitness']['mean']
            )
            
            for algo_name, stats in sorted_results:
                print(f"{algo_name:<25} "
                      f"{stats['fitness']['mean']:<15.6f} "
                      f"{stats['fitness']['std']:<12.6f} "
                      f"{stats['fitness']['min']:<12.6f} "
                      f"{stats['time']['mean']:<10.4f}")
            
            # Best algorithm
            best_algo = sorted_results[0][0]
            best_fitness = sorted_results[0][1]['fitness']['mean']
            print(f"\n  🏆 Best: {best_algo} (fitness={best_fitness:.6f})")


def main():
    """
    Main function để chạy experiments.
    """
    
    # 1. Định nghĩa các bài toán
    print("Setting up problems...")
    problems = [
        SphereFunction(dim=10),
        RastriginFunction(dim=10),
        # RosenbrockFunction(dim=10),
        # AckleyFunction(dim=10),
    ]
    
    # 2. Định nghĩa các thuật toán
    print("Setting up algorithms...")
    algorithms = [
        FireflyAlgorithm(population_size=30, beta0=1.0, gamma=1.0, alpha=0.2),
        # PSO(population_size=30, w=0.7, c1=1.5, c2=1.5),
        # ABC(n_bees=40),
        HillClimbing(step_size=0.1),
    ]
    
    # 3. Tạo experiment
    experiment = ContinuousExperiment(
        algorithms=algorithms,
        problems=problems,
        n_runs=10, 
        max_iter=100
    )
    
    # 4. Chạy experiments
    experiment.run_all_experiments()
    
    # 5. In tóm tắt
    experiment.print_summary()
    
    # 6. Lưu kết quả
    experiment.save_results()


if __name__ == "__main__":
    main()
