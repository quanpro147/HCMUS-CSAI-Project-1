import numpy as np
import json
import time
from datetime import datetime
from typing import List, Dict, Any
import os
from utils import compute_basic_stats, compute_convergence_speed, compute_time_complexity, compute_robustness_metrics

from problems import (
    SphereFunction, 
    RastriginFunction, 
    RosenbrockFunction,
    AckleyFunction
)

from algorithms import (
    FireflyAlgorithm,
    ParticleSwarmOptimization,
    ArtificialBeeColony,
    HillClimbing,
)


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
    
    def run(self):

        print("RUNNING CONTINUOUS OPTIMIZATION EXPERIMENTS")
        print("="*80)
        
        for problem in self.problems:
            print("\n" + "-"*80)
            print(f"Problem: {problem.prob_name} (dim={problem.dim})")
            print("-"*80)

            problem_results = {}

            for algorithm in self.algorithms:
                stats = self.run_multiple_experiments(algorithm, problem)
                problem_results[algorithm.name] = stats
            
            self.results[problem.prob_name] = problem_results

        self.summary()
        self.save_results()
        
    def save_results(self, filename: str = None):

        if filename is None:
            filename = f"continuous_results.json"
        
        filepath = os.path.join(self.results_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.results, f, indent=4)
            print(f"\nResults saved to: {filepath}")
        except Exception as e:
            print(f"\nError saving results to {filepath}: {e}")
    
    def summary(self):

        print("EXPERIMENT SUMMARY")
        
        for problem_name, problem_results in self.results.items():
            print(f"\n\nProblem: {problem_name}")
            print(f"{'Algorithm':<25} {'Mean Fitness':<15} {'Std Fitness':<12} {'Best Fitness':<12} {'Mean Time (s)':<10}")
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
            print(f"\n  Best: {best_algo} (fitness={best_fitness:.6f})")


def main():

    problems = [
        SphereFunction(dim=10),
        RastriginFunction(dim=10),
        RosenbrockFunction(dim=10),
        AckleyFunction(dim=10),
    ]
    
    algorithms = [
        FireflyAlgorithm(),
        ParticleSwarmOptimization(),
        ArtificialBeeColony(),
        HillClimbing(),
    ]

    experiment = ContinuousExperiment(
        algorithms=algorithms,
        problems=problems,
        n_runs=10, 
        max_iter=100
    )
    
    experiment.run()


if __name__ == "__main__":
    main()
