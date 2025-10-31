import sys
import os
from datetime import datetime

# Import problems
from problems import (
    SphereFunction,
    RastriginFunction,
    RosenbrockFunction,
    AckleyFunction,
)

# Import algorithms
from algorithms import (
    ParticleSwarmOptimization,
    ArtificialBeeColony,
    AntColonyOptimization,
    FireflyAlgorithm,
    CuckooSearch,
    HillClimbing,
    # AStar,
    # BFS,
)

# Import experiment class
from experiments.run_continuous_tests import ContinuousExperiment
# from experiments.run_discrete_tests import DiscreteExperiment

def print_banner():
    print("\n" + "="*80)
    print("    🔬 ALGORITHM COMPARISON EXPERIMENTS")
    print("    Swarm Intelligence vs Traditional Search")
    print("="*80)

def setup_problems(dimensions=[10, 30]):
    """
    Thiết lập các bài toán test.
    
    Args:
        dimensions: List các số chiều cần test
    """
    print("Setting up problems...")
    
    problems = []
    
    for dim in dimensions:
        problems.extend([
            SphereFunction(dim=dim),
            RastriginFunction(dim=dim),
            #RosenbrockFunction(dim=dim),
            #AckleyFunction(dim=dim),
        ])
    
    print(f"Created {len(problems)} problems")
    for p in problems:
        print(f"    - {p.prob_name}")
    
    return problems


def setup_algorithms():
    """Thiết lập các thuật toán."""
    print("\nSetting up algorithms...")
    
    algorithms = [
        # ParticleSwarmOptimization(population_size=30, w=0.7, c1=1.5, c2=1.5),
        # ArtificialBeeColony(population_size=40, limit=50),
        # AntColonyOptimization(n_ants=30),
        FireflyAlgorithm(population_size=30, beta0=1.0, gamma=1.0, alpha=0.2),
        CuckooSearch(population_size=25, pa=0.25),
        HillClimbing(max_neighbors=10, step_size=0.1),
    ]
    
    print(f"Created {len(algorithms)} algorithms")
    for algo in algorithms:
        print(f"    - {algo.name}")
    
    return algorithms


def create_experiment(problems, algorithms, n_runs=30, max_iter=100):
    """
    Tạo experiment object.
    
    Args:
        problems: List các bài toán
        algorithms: List các thuật toán
        n_runs: Số lần chạy mỗi experiment
        max_iter: Số iteration tối đa
    
    Returns:
        ContinuousExperiment object
    """
    print("\nCreating experiment...")
    print(f"   Problems: {len(problems)}")
    print(f"   Algorithms: {len(algorithms)}")
    print(f"   Runs per experiment: {n_runs}")
    print(f"   Max iterations: {max_iter}")
    print(f"   Total runs: {len(problems) * len(algorithms) * n_runs}")
    
    experiment = ContinuousExperiment(
        algorithms=algorithms,
        problems=problems,
        n_runs=n_runs,
        max_iter=max_iter,
        results_dir="results"
    )
    
    return experiment


def main():
    # 1. Print banner
    print_banner()
    
    # 2. Configuration
    print("Configuration:")
    CONFIG = {
        'dimensions': [10],      
        'n_runs': 10,
        'max_iter': 100,
    }
    
    for key, value in CONFIG.items():
        print(f" {key}: {value}")
    print()
    
    # 3. Setup
    problems = setup_problems(dimensions=CONFIG['dimensions'])
    algorithms = setup_algorithms()

    # 4. Create experiment
    experiment = create_experiment(
        problems=problems,
        algorithms=algorithms,
        n_runs=CONFIG['n_runs'],
        max_iter=CONFIG['max_iter']
    )
    
    # 5. Run experiment
    try:
        experiment.run()
        print(f"✅ ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")

    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
