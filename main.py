import sys
import os
from datetime import datetime

# ==== Import continuous problems ====
from problems import (
    SphereFunction,
    RastriginFunction,
    RosenbrockFunction,
    AckleyFunction,
)

# ==== Import discrete problems ====
from problems import (
    TravelingSalesmanProblem,
    KnapsackProblem,
)

# ==== Import algorithms ====
from algorithms import (
    ParticleSwarmOptimization,
    ArtificialBeeColony,
    AntColonyOptimization,
    FireflyAlgorithm,
    CuckooSearch,
    HillClimbing,
    SimulatedAnnealing,
    AStar,
    BFS,
)

# ==== Import experiment classes ====
from experiments.run_continuous_tests import ContinuousExperiment
from experiments.run_discrete_tests import DiscreteExperiment


# ================================================================
#                    COMMON UTILITIES
# ================================================================

def print_banner():
    print("\n" + "=" * 80)
    print("    🔬 ALGORITHM COMPARISON EXPERIMENTS")
    print("    Swarm Intelligence & Classical Search Methods")
    print("=" * 80)


# ================================================================
#                    CONTINUOUS EXPERIMENTS
# ================================================================

def setup_continuous_problems(dimensions=[10]):
    """Thiết lập các bài toán liên tục."""
    print("\n📈 Setting up continuous problems...")
    problems = []
    for dim in dimensions:
        problems.extend([
            SphereFunction(dim=dim),
            RastriginFunction(dim=dim),
            # RosenbrockFunction(dim=dim),
            # AckleyFunction(dim=dim),
        ])
    print(f"→ Created {len(problems)} continuous problems")
    for p in problems:
        print(f"    - {p.prob_name}")
    return problems


def setup_continuous_algorithms():
    """Thiết lập các thuật toán continuous."""
    print("\n⚙️  Setting up continuous algorithms...")
    algorithms = [
        ParticleSwarmOptimization(),  # Sử dụng tham số từ config
        ArtificialBeeColony(),
        FireflyAlgorithm(),
        CuckooSearch(),
        HillClimbing(),
    ]
    print(f"→ Created {len(algorithms)} continuous algorithms")
    for algo in algorithms:
        print(f"    - {algo.name}")
    return algorithms


def run_continuous_experiments(config):
    problems = setup_continuous_problems(dimensions=config['dimensions'])
    algorithms = setup_continuous_algorithms()

    experiment = ContinuousExperiment(
        algorithms=algorithms,
        problems=problems,
        n_runs=config['n_runs'],
        max_iter=config['max_iter'],
        results_dir="results/continuous"
    )

    experiment.run()
    print("\n✅ Continuous experiments completed!\n")


# ================================================================
#                    DISCRETE EXPERIMENTS
# ================================================================

def setup_discrete_problems():
    """Thiết lập các bài toán rời rạc."""
    print("\n🧩 Setting up discrete problems...")
    problems = [
        TravelingSalesmanProblem(n_cities=10),
        #KnapsackProblem(n_items=20, capacity=50),
    ]
    print(f"→ Created {len(problems)} discrete problems")
    for p in problems:
        print(f"    - {p.prob_name}")
    return problems


def setup_discrete_algorithms():
    """Thiết lập các thuật toán discrete."""
    print("\n⚙️  Setting up discrete algorithms...")
    algorithms = [
        AntColonyOptimization(),     # Swarm-based cho TSP
        SimulatedAnnealing(),        # Probabilistic local search
        BFS(),                       # Complete search
        # AStar(),                   # Chỉ dùng cho pathfinding (GridPathfinding)
    ]
    print(f"→ Created {len(algorithms)} discrete algorithms")
    for algo in algorithms:
        print(f"    - {algo.name}")
    return algorithms


def run_discrete_experiments(config):
    problems = setup_discrete_problems()
    algorithms = setup_discrete_algorithms()

    experiment = DiscreteExperiment(
        algorithms=algorithms,
        problems=problems,
        n_runs=config['n_runs'],
        max_iter=config['max_iter'],
        results_dir="results/discrete"
    )

    experiment.run()
    print("\n✅ Discrete experiments completed!\n")


# ================================================================
#                    MAIN EXECUTION
# ================================================================

def main():
    print_banner()

    # --- Configuration ---
    CONFIG = {
        'dimensions': [10],
        'n_runs': 5,
        'max_iter': 100,
    }

    print("\nConfiguration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    print()

    try:
        # Chạy cả 2 loại experiment
        #run_continuous_experiments(CONFIG)
        run_discrete_experiments(CONFIG)

        print("\n🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")

    except KeyboardInterrupt:
        print("\n⚠️  Experiment interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
