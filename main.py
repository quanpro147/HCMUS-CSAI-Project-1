# Tên file: main.py

import sys
import os
from datetime import datetime

# Import config
from config import EXPERIMENT_CONFIG

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
    FireflyAlgorithm,
    CuckooSearch,
    HillClimbing,
)

# Import experiment runners
from experiments.run_continuous_tests import ContinuousExperiment
from experiments.run_discrete_tests import DiscreteExperiment # <-- THÊM DÒNG NÀY

def print_banner():
    print("\n" + "="*80)
    print("    🔬 ALGORITHM COMPARISON EXPERIMENTS")
    print("    Swarm Intelligence vs Traditional Search")
    print("="*80)

def run_all_experiments():
    """
    Hàm chính điều phối việc chạy tất cả thí nghiệm.
    """
    # 1. Lấy cấu hình chung
    n_runs = EXPERIMENT_CONFIG.get('n_runs', 10)
    max_iter = EXPERIMENT_CONFIG.get('max_iter', 100)
    results_dir = EXPERIMENT_CONFIG.get('results_dir', 'results')
    
    # 2. Chạy Thí nghiệm Liên tục (Continuous)
    # print("\n" + "="*80)
    # print("    🔬 BẮT ĐẦU CHẠY CONTINUOUS EXPERIMENTS")
    # print("="*80)
    
    # # Setup problems
    # cont_problems = []
    # cont_dims = EXPERIMENT_CONFIG.get('continuous_dims', [10])
    # for dim in cont_dims:
    #     cont_problems.extend([
    #         SphereFunction(dim=dim),
    #         RastriginFunction(dim=dim),
    #         # Thêm các hàm khác ở đây
    #     ])

    # # Setup algorithms
    # cont_algorithms = [
    #     ParticleSwarmOptimization(), # Đọc params từ config
    #     ArtificialBeeColony(),       # Đọc params từ config
    #     FireflyAlgorithm(),          # Đọc params từ config
    #     CuckooSearch(),              # Đọc params từ config
    #     HillClimbing(),              # Đọc params từ config
    # ]
    
    # # Tạo và chạy
    # cont_experiment = ContinuousExperiment(
    #     algorithms=cont_algorithms,
    #     problems=cont_problems,
    #     n_runs=n_runs,
    #     max_iter=max_iter,
    #     results_dir=results_dir
    # )
    # cont_experiment.run()
    
    # 3. Chạy Thí nghiệm Rời rạc (Discrete)
    # (File này tự đọc config và setup bên trong)
    disc_experiment = DiscreteExperiment(
        n_runs=n_runs,
        results_dir=results_dir
    )
    disc_experiment.run()

def main():
    # 1. In banner
    print_banner()
    
    try:
        # 2. Chạy tất cả
        run_all_experiments()
        print("\n" + "="*80)
        print("✅ ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("="*80)

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