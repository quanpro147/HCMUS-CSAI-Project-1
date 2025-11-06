
"""
Configuration file cho experiments.
Chỉnh sửa các tham số ở đây thay vì sửa code.
"""

# Experiment Configuration
EXPERIMENT_CONFIG = {
    # Số lần chạy mỗi thuật toán trên mỗi bài toán (cho Robustness)
    'n_runs': 10,  # Bắt đầu với 10, sau tăng lên 30
    
    # Số iteration tối đa cho mỗi thuật toán
    'max_iter': 100,
    
    # Thư mục lưu kết quả
    'results_dir': 'results',
    
    # Cấu hình cho Scalability (Continuous)
    'continuous_dims': [10, 30], # Test 10 và 30 chiều
    
    # Cấu hình cho Scalability (Discrete)
    'tsp_sizes': [10, 20], # Test TSP 10 và 20 thành phố
    'grid_sizes': [ (10,10) ], # Test lưới 10x10
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
    
    'aco': { # Dùng cho ACO-TSP
        'n_ants': 30,
        'alpha': 1.0,     # Pheromone importance
        'beta': 2.0,      # Heuristic importance
        'rho': 0.1,       # Evaporation rate
        'pheromone_scale': 100,
    },
    
    'aco_pathfinder': { # Dùng cho ACO-Grid
        'n_ants': 30,
        'alpha': 1.0,
        'beta': 5.0,      # Heuristic (beta) quan trọng hơn cho tìm đường
        'rho': 0.1,
        'pheromone_scale': 100,
    },
    
    'a_star': {
        'max_iter': 50000 # Giới hạn số nút A* được phép duyệt
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

# (Các list PROBLEMS_TO_TEST và ALGORITHMS_TO_TEST không còn cần thiết)
# Chúng ta sẽ định nghĩa experiment trực tiếp trong file runner