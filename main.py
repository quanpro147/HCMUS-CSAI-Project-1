import sys
from config import EXPERIMENT_CONFIG
from testcases_loader import load_testcases

from algorithms import (
    ParticleSwarmOptimization,
    ArtificialBeeColony,
    FireflyAlgorithm,
    CuckooSearch,
    HillClimbing,
    AntColonyOptimization,
    ACO_Pathfinder,
    AStar,
    SimulatedAnnealing,
    BFS
)

from experiments.run_continuous_tests import ContinuousExperiment
from experiments.run_discrete_tests import DiscreteExperiment


def run_all_experiments():

    # 1. Get common configuration
    n_runs = EXPERIMENT_CONFIG.get('n_runs', 10)
    max_iter = EXPERIMENT_CONFIG.get('max_iter', 100)
    results_dir = EXPERIMENT_CONFIG.get('results_dir', 'results')
    
    # 2. Run continuous experiments
    cont_problems = load_testcases("testcases/continuous_testcases.json")
    
    cont_algorithms = [
        ParticleSwarmOptimization(),
        ArtificialBeeColony(),
        FireflyAlgorithm(),
        CuckooSearch(),
        HillClimbing(),
    ]

    cont_experiment = ContinuousExperiment(
        algorithms=cont_algorithms,
        problems=cont_problems,
        n_runs=n_runs, 
        max_iter=max_iter,
        results_dir=results_dir
    )
    
    #cont_experiment.run()
    
    # 3. Run discrete experiments
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

def main():

    print("\n" + "=" * 80)
    print("    ALGORITHM COMPARISON EXPERIMENTS")
    print("    Swarm Intelligence & Classical Search Methods")
    print("=" * 80)

    try:
        run_all_experiments()
        print("\n" + "="*80)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("="*80)

    except KeyboardInterrupt:
        print("\n  Experiment interrupted by user.")
        sys.exit(1)

    except Exception as e:
        print(f"\n Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()