"""
Configuration file cho experiments.

"""

EXPERIMENT_CONFIG = {
    # Số lần chạy mỗi thuật toán trên mỗi bài toán
    'n_runs': 10,

    # Số iteration tối đa cho mỗi thuật toán
    'max_iter': 100,

    # Các số chiều cần test
    'dimensions': [10, 30],

    # Thư mục lưu kết quả
    'results_dir': 'results',
    
    # Cấu hình cho Scalability (Continuous)
    'continuous_dims': [10, 30], # Test 10 và 30 chiều
    
    # Cấu hình cho Scalability (Discrete)
    'tsp_sizes': [10, 20], # Test TSP 10 và 20 thành phố
    'grid_sizes': [ (10,10) ], # Test lưới 10x10
}

# ============================================================
# ⚙️ ALGORITHM PARAMETERS
# ============================================================

ALGORITHM_PARAMS = {
    # 🐦 Particle Swarm Optimization
    'pso': {
        'population_size': 30,
        'w': 0.8,         # inertia cao hơn để tránh local minima
        'c1': 1.6,        # cognitive
        'c2': 1.6,        # social
    },

    # -------------------------
    # 🐝 Artificial Bee Colony
    # -------------------------
    'abc': {
        'population_size': 40,
        'limit': 40,     
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
        'population_size': 40,  # tăng quần thể để cải thiện đa dạng
        'beta0': 1.0,           # attractiveness
        'gamma': 0.1,           # giảm hấp thụ ánh sáng để firefly di chuyển xa hơn
        'alpha': 0.25,          # tăng chút randomization để tránh local minima
    },

    # -------------------------
    # 🥚 Cuckoo Search
    # -------------------------
    'cs': {
        'population_size': 30,
        'pa': 0.15,        # giảm xác suất bỏ tổ (đỡ mất cá thể tốt)
        'alpha': 0.005,    # giảm bước Levy flight để tránh nhảy quá xa
    },

    # -------------------------
    # 🐜 Ant Colony Optimization
    # -------------------------
    'aco': {
        'population_size': 20,     # số lượng kiến
        'alpha': 1.0,              # hệ số quan trọng của pheromone
        'beta': 2.0,               # hệ số quan trọng của khoảng cách
        'evaporation': 0.5,        # tỷ lệ bay hơi pheromone
        'pheromone_scale': 100,    # hệ số Q trong công thức cập nhật
    },

    # Hill Climbing
    'hill_climbing': {
        'max_neighbors': 20,  # tăng số hàng xóm để cải thiện tìm kiếm
        'step_size': 0.05,    # bước nhỏ hơn giúp chính xác hơn
    },

    # -------------------------
    # 🔥 Simulated Annealing
    # -------------------------
    'simulated_annealing': {
        'initial_temp': 1000,      # nhiệt độ ban đầu
        'cooling_rate': 0.95,      # tỷ lệ làm nguội (0.95-0.99)
        'min_temp': 1e-3,          # nhiệt độ tối thiểu
    },
}

# PROBLEMS

PROBLEMS_TO_TEST = {
    'continuous': [
        'sphere',
        'rastrigin',
        'rosenbrock',
        'ackley',
    ],

    'discrete': [
        'tsp',
        'knapsack',
    ],
}

# ALGORITHMS TO TEST

ALGORITHMS_TO_TEST = {
    'swarm': [
        'pso',
        'abc',
        'aco',
        'fa',
        'cs',
    ],

    'traditional': [
        'hill_climbing',
        'bfs',
        'a_star',
        'simulated_annealing',
    ],
}
