import numpy as np
import json
import time
from datetime import datetime
from typing import List, Dict, Any
import os

# Import problems
from problems import (
    TravelingSalesmanProblem,
    KnapsackProblem,
)

# Import algorithms
from algorithms import AntColonyOptimization
from algorithms import BFS
from algorithms import AStar


class DiscreteExperiment:
    """
    Class để chạy và quản lý experiments cho các bài toán rời rạc
    (như TSP, Knapsack, v.v).
    """

    def __init__(self,
                 algorithms: List,
                 problems: List,
                 n_runs: int = 30,
                 max_iter: int = 100,
                 results_dir: str = "results"):
        self.algorithms = algorithms
        self.problems = problems
        self.n_runs = n_runs
        self.max_iter = max_iter
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.results = {}

    def run_single_experiment(self, algorithm, problem, run_id: int) -> Dict[str, Any]:
        """
        Chạy 1 lần experiment cho 1 thuật toán trên 1 bài toán.
        """
        print(f"    Run {run_id + 1}/{self.n_runs}...", end='')

        algorithm.reset() if hasattr(algorithm, "reset") else None

        start = time.time()
        result = algorithm.run(problem, max_iter=self.max_iter)
        exec_time = time.time() - start

        # Chuẩn hóa output nếu thuật toán không trả execution_time
        if "execution_time" not in result:
            result["execution_time"] = exec_time

        print(f" fitness={result.get('fitness', np.nan):.4f}, time={result['execution_time']:.4f}s")
        return result

    def run_algorithm_on_problem(self, algorithm, problem) -> Dict[str, Any]:
        """
        Chạy 1 thuật toán nhiều lần trên 1 bài toán.
        """
        print(f"\nTesting {algorithm.name} on {problem.prob_name}...")

        all_results = []
        for run_id in range(self.n_runs):
            result = self.run_single_experiment(algorithm, problem, run_id)
            all_results.append(result)

        fitness_values = [r["fitness"] for r in all_results if "fitness" in r]
        time_values = [r["execution_time"] for r in all_results]
        iterations_values = [r.get("iterations", np.nan) for r in all_results]

        # Get best solution and convert to list if it's a numpy array
        best_solution = all_results[np.argmin(fitness_values)].get("solution", None)
        if isinstance(best_solution, np.ndarray):
            best_solution = best_solution.tolist()

        stats = {
            "algorithm": algorithm.name,
            "problem": problem.prob_name,
            "n_runs": self.n_runs,
            "max_iter": self.max_iter,

            # Fitness statistics
            "fitness": {
                "mean": float(np.mean(fitness_values)),
                "std": float(np.std(fitness_values)),
                "min": float(np.min(fitness_values)),
                "max": float(np.max(fitness_values)),
                "median": float(np.median(fitness_values)),
                "q25": float(np.percentile(fitness_values, 25)),
                "q75": float(np.percentile(fitness_values, 75)),
                "all_values": [float(v) for v in fitness_values],  # Convert to list of floats
            },

            # Time statistics
            "time": {
                "mean": float(np.mean(time_values)),
                "std": float(np.std(time_values)),
                "min": float(np.min(time_values)),
                "max": float(np.max(time_values)),
                "all_values": [float(t) for t in time_values],  # Convert to list of floats
            },

            # Iterations
            "iterations": {
                "mean": float(np.nanmean(iterations_values))
            },

            # Best solution
            "best_solution": best_solution,
        }

        return stats

    def run_all_experiments(self):
        """
        Chạy tất cả experiments: tất cả thuật toán trên tất cả bài toán.
        """
        print("RUNNING DISCRETE OPTIMIZATION EXPERIMENTS")
        print(f"Number of algorithms: {len(self.algorithms)}")
        print(f"Number of problems: {len(self.problems)}")
        print(f"Runs per experiment: {self.n_runs}")
        print(f"Max iterations: {self.max_iter}")

        start_time = time.time()

        for problem in self.problems:
            print(f"\nProblem: {problem.prob_name}")
            problem_results = {}

            for algorithm in self.algorithms:
                stats = self.run_algorithm_on_problem(algorithm, problem)
                problem_results[algorithm.name] = stats

            self.results[problem.prob_name] = problem_results

        total_time = time.time() - start_time
        print(f"\nALL EXPERIMENTS COMPLETED in {total_time:.2f}s")

    def print_summary(self):
        """
        In tóm tắt kết quả.
        """
        print("\nEXPERIMENT SUMMARY")
        for problem_name, problem_results in self.results.items():
            print(f"\n {problem_name}:")
            print(f"{'Algorithm':<25} {'Mean Fitness':<15} {'Std':<12} {'Best':<12} {'Time (s)':<10}")
            print("-" * 80)

            sorted_results = sorted(
                problem_results.items(),
                key=lambda x: x[1]["fitness"]["mean"]
            )

            for algo_name, stats in sorted_results:
                print(f"{algo_name:<25} "
                      f"{stats['fitness']['mean']:<15.6f} "
                      f"{stats['fitness']['std']:<12.6f} "
                      f"{stats['fitness']['min']:<12.6f} "
                      f"{stats['time']['mean']:<10.4f}")

            best_algo = sorted_results[0][0]
            best_fitness = sorted_results[0][1]["fitness"]["mean"]
            print(f"\n  🏆 Best: {best_algo} (fitness={best_fitness:.6f})")

    def save_results(self, filename: str = None):
        """
        Lưu kết quả ra file JSON.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"discrete_results_{timestamp}.json"

        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=4)

        print(f"\nResults saved to: {filepath}")
        return filepath

    def run(self):
        """
        Pipeline chính: chạy toàn bộ experiments -> in summary -> lưu kết quả
        """
        self.run_all_experiments()
        self.print_summary()
        return self.save_results()


def main():
    """
    Main function để chạy toàn bộ discrete experiments.
    """

    print("Setting up problems...")
    problems = [
        TravelingSalesmanProblem(n_cities=20),
        KnapsackProblem(n_items=30, capacity=100),
    ]

    print("Setting up algorithms...")
    algorithms = [
        AntColonyOptimization(n_ants=30, alpha=1.0, beta=2.0, rho=0.5),
        AStar(),
        BFS(),
    ]

    experiment = DiscreteExperiment(
        algorithms=algorithms,
        problems=problems,
        n_runs=5,
        max_iter=100,
        results_dir="results_discrete"
    )

    experiment.run()


if __name__ == "__main__":
    main()
