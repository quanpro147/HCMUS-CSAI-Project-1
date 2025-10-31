"""
Configuration file cho experiments.
Chỉnh sửa các tham số ở đây thay vì sửa code.
"""

# Experiment Configuration
EXPERIMENT_CONFIG = {
    # Số lần chạy mỗi thuật toán trên mỗi bài toán
    # Khuyến nghị: 30 để có statistical significance
    'n_runs': 10,  # Bắt đầu với 10, sau tăng lên 30
    
    # Số iteration tối đa cho mỗi thuật toán
    'max_iter': 100,
    
    # Các số chiều cần test
    'dimensions': [10],  # Có thể thêm [10, 30, 50]
    
    # Thư mục lưu kết quả
    'results_dir': 'results',
}

# Algorithm Parameters
ALGORITHM_PARAMS = {
    'pso': {
        'population_size': 30,
        'w': 0.7,         # Inertia weight
        'c1': 1.5,        # Cognitive coefficient
        'c2': 1.5,        # Social coefficient
    },
    
    'abc': {
        'population_size': 40,
        'limit': 50,      # Abandonment limit
    },
    
    'aco': {
        'n_ants': 30,
        'alpha': 1.0,     # Pheromone importance
        'beta': 2.0,      # Heuristic importance
        'rho': 0.1,       # Evaporation rate
    },
    
    'fa': {
        'population_size': 30,
        'beta0': 1.0,     # Attractiveness
        'gamma': 1.0,     # Light absorption
        'alpha': 0.2,     # Randomization
    },
    
    'cs': {
        'population_size': 25,
        'pa': 0.25,       # Probability of abandon
    },
    
    'hill_climbing': {
        'max_neighbors': 10,
        'step_size': 0.1,
    },
}

# Problems to test
PROBLEMS_TO_TEST = {
    'continuous': [
        'sphere',
        'rastrigin',
        'rosenbrock',
        'ackley',
        # 'griewank',
        # 'schwefel',
    ],
    
    'discrete': [
        # 'tsp',
        # 'knapsack',
    ],
}

# Algorithms to test
ALGORITHMS_TO_TEST = {
    'swarm': [
        'pso',
        'abc',
        # 'aco',
        'fa',
        'cs',
    ],
    
    'traditional': [
        'hill_climbing',
        # 'simulated_annealing',
    ],
}
