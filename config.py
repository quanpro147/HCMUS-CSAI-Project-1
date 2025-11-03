"""
Configuration file cho experiments.

"""

EXPERIMENT_CONFIG = {
    # Số lần chạy mỗi thuật toán trên mỗi bài toán
    'n_runs': 30,

    # Số iteration tối đa cho mỗi thuật toán
    'max_iter': 300,

    # Các số chiều cần test
    'dimensions': [10, 30],

    # Thư mục lưu kết quả
    'results_dir': 'results',
}

# ============================================================
# ⚙️ ALGORITHM PARAMETERS (tối ưu theo benchmark)
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
        'limit': 40,      # giảm nhẹ để tránh stagnation
    },

    # -------------------------
    # 🔥 Firefly Algorithm
    # -------------------------
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
        # 'knapsack',
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
    ],
}
