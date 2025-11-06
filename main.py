import sys
from config import EXPERIMENT_CONFIG

from problems import (
    SphereFunction,
    RastriginFunction,
    #RosenbrockFunction,
    AckleyFunction,
)

from algorithms import (
    ParticleSwarmOptimization,
    ArtificialBeeColony,
    FireflyAlgorithm,
    CuckooSearch,
    HillClimbing,
)

from experiments.run_continuous_tests import ContinuousExperiment
from experiments.run_discrete_tests import DiscreteExperiment


def run_all_experiments():

    # 1. Get common configuration
    n_runs = EXPERIMENT_CONFIG.get('n_runs', 10)
    max_iter = EXPERIMENT_CONFIG.get('max_iter', 100)
    results_dir = EXPERIMENT_CONFIG.get('results_dir', 'results')
    
    # 2. Run continuous experiments
    cont_problems = [
        SphereFunction(dim=10),
        RastriginFunction(dim=10),
        #RosenbrockFunction(dim=10),
        AckleyFunction(dim=10),
    ]

    cont_algorithms = [
        ParticleSwarmOptimization(), 
        ArtificialBeeColony(),       
        FireflyAlgorithm(),          
        CuckooSearch(),              
        HillClimbing(),              
    ]
    
    # Create and run continuous experiment
    cont_experiment = ContinuousExperiment(
        algorithms=cont_algorithms,
        problems=cont_problems,
        n_runs=n_runs,
        max_iter=max_iter,
        results_dir=results_dir
    )
    cont_experiment.run()
    
    # Run discrete experiments
    disc_experiment = DiscreteExperiment(
        n_runs=n_runs,
        max_iter=max_iter,
        results_dir=results_dir
    )
    disc_experiment.run()

def main():

    print("\n" + "=" * 80)
    print("    ALGORITHM COMPARISON EXPERIMENTS")
    print("    Swarm Intelligence & Classical Search Methods")
    print("=" * 80)

    try:
        # 2. Run all experiments
        run_all_experiments()
        print("\n" + "="*80)
        print("✅ ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("="*80)

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