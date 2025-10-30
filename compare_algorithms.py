"""
Demo: So sánh thuật toán Swarm vs Traditional trên cùng bài toán.
"""

import numpy as np
from problems import SphereFunction, RastriginFunction
from algorithms.swarm_intelligence.pso_example import PSO
# from algorithms.swarm_intelligence.abc import ABC
# from algorithms.traditional_search.hill_climbing import HillClimbing


def compare_algorithms_on_problem(problem, algorithms, n_runs=10, max_iter=100):
    """
    So sánh nhiều thuật toán trên cùng một bài toán.
    
    Args:
        problem: Bài toán cần test
        algorithms: List các thuật toán
        n_runs: Số lần chạy mỗi thuật toán
        max_iter: Số iteration tối đa
    """
    print("=" * 80)
    print(f"COMPARING ALGORITHMS ON: {problem.name}")
    print("=" * 80)
    print(f"Dimension: {problem.dim}")
    print(f"Optimal value: {problem.optimal_value}")
    print(f"Number of runs: {n_runs}")
    print(f"Max iterations: {max_iter}")
    print()
    
    results = {}
    
    for algo in algorithms:
        print(f"\nRunning {algo.name}...")
        
        fitness_list = []
        time_list = []
        evaluations_list = []
        
        for run in range(n_runs):
            # Chạy thuật toán
            result = algo.run(problem, max_iter=max_iter)
            
            fitness_list.append(result['fitness'])
            time_list.append(result['execution_time'])
            evaluations_list.append(result['function_evaluations'])
            
            if (run + 1) % 5 == 0:
                print(f"  Run {run + 1}/{n_runs}: fitness = {result['fitness']:.6f}")
        
        # Lưu kết quả
        results[algo.name] = {
            'fitness': {
                'mean': np.mean(fitness_list),
                'std': np.std(fitness_list),
                'min': np.min(fitness_list),
                'max': np.max(fitness_list),
                'median': np.median(fitness_list)
            },
            'time': {
                'mean': np.mean(time_list),
                'std': np.std(time_list)
            },
            'evaluations': {
                'mean': np.mean(evaluations_list)
            },
            'raw_fitness': fitness_list,
            'raw_time': time_list
        }
    
    return results


def print_comparison_table(results, problem):
    """In bảng so sánh kết quả."""
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    # Header
    print(f"{'Algorithm':<20} {'Mean Fitness':<15} {'Std':<12} {'Min':<12} {'Time (s)':<12}")
    print("-" * 80)
    
    # Sort by mean fitness
    sorted_algos = sorted(results.items(), key=lambda x: x[1]['fitness']['mean'])
    
    for algo_name, stats in sorted_algos:
        print(f"{algo_name:<20} "
              f"{stats['fitness']['mean']:<15.6f} "
              f"{stats['fitness']['std']:<12.6f} "
              f"{stats['fitness']['min']:<12.6f} "
              f"{stats['time']['mean']:<12.4f}")
    
    print()
    
    # Tìm thuật toán tốt nhất
    best_algo = sorted_algos[0][0]
    best_fitness = sorted_algos[0][1]['fitness']['mean']
    
    print(f"🏆 Best Algorithm: {best_algo}")
    print(f"   Mean Fitness: {best_fitness:.6f}")
    print(f"   Error from optimal: {abs(best_fitness - problem.optimal_value):.6f}")
    
    # So sánh với optimal
    print("\n📊 Error from Optimal Value:")
    for algo_name, stats in sorted_algos:
        error = abs(stats['fitness']['mean'] - problem.optimal_value)
        error_pct = (error / max(abs(problem.optimal_value), 1)) * 100
        print(f"   {algo_name:<20}: {error:.6f} ({error_pct:.2f}%)")


def compare_swarm_vs_traditional():
    """So sánh Swarm Intelligence vs Traditional Search."""
    
    # Tạo bài toán
    problem = SphereFunction(dim=10)
    
    # Danh sách thuật toán (MIX swarm và traditional)
    algorithms = [
        PSO(population_size=30, w=0.7, c1=1.5, c2=1.5),
        # ABC(n_bees=40),  # Swarm
        # ACO(n_ants=30),  # Swarm
        # HillClimbing(step_size=0.1),  # Traditional
        # BFS(),  # Traditional (nếu áp dụng được)
    ]
    
    # Chạy so sánh
    results = compare_algorithms_on_problem(
        problem=problem,
        algorithms=algorithms,
        n_runs=10,
        max_iter=100
    )
    
    # In kết quả
    print_comparison_table(results, problem)
    
    return results


def compare_on_multiple_problems():
    """So sánh trên nhiều bài toán khác nhau."""
    
    problems = [
        SphereFunction(dim=10),
        RastriginFunction(dim=10),
        # AckleyFunction(dim=10),
    ]
    
    algorithms = [
        PSO(population_size=30),
        # ABC(n_bees=40),
        # HillClimbing(step_size=0.1),
    ]
    
    print("\n" + "=" * 80)
    print("COMPARING ON MULTIPLE PROBLEMS")
    print("=" * 80)
    
    all_results = {}
    
    for problem in problems:
        print(f"\n\n{'='*80}")
        print(f"Problem: {problem.name}")
        print(f"{'='*80}")
        
        results = compare_algorithms_on_problem(
            problem=problem,
            algorithms=algorithms,
            n_runs=5,
            max_iter=50
        )
        
        all_results[problem.name] = results
        print_comparison_table(results, problem)
    
    return all_results


if __name__ == "__main__":
    print("🔬 ALGORITHM COMPARISON DEMO")
    print("Comparing Swarm Intelligence vs Traditional Search")
    print()
    
    # So sánh trên 1 bài toán
    results = compare_swarm_vs_traditional()
    
    # Uncomment để so sánh trên nhiều bài toán
    # all_results = compare_on_multiple_problems()
    
    print("\n✅ Comparison completed!")
