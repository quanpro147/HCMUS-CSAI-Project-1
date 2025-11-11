EXPERIMENT_CONFIG = {
    'n_runs': 10,
    'max_iter': 100,
    'dimensions': [10, 30],
    'results_dir': 'results',
    'continuous_dims': [10, 30],
    'tsp_sizes': [10, 20],
    'grid_sizes': [ (10,10) ],
}

# ALGORITHM PARAMETERS
ALGORITHM_PARAMS = {
    'pso': {
        'population_size': 30,
        'w': 0.8,
        'c1': 1.6,
        'c2': 1.6,
    },
    'abc': {
        'population_size': 40,
        'limit': 40,
    },
    'aco': {
        'n_ants': 20,
        'alpha': 1.0,
        'beta': 2.0,
        'evaporation': 0.5,
        'pheromone_scale': 100,
    },
    'aco_pathfinder': {
        'n_ants': 30,
        'alpha': 1.0,
        'beta': 5.0,
        'rho': 0.1,
        'pheromone_scale': 100,
    },
    'a_star': {
        'max_iter': 50000
    },
    'fa': {
        'population_size': 40,
        'beta0': 1.0,
        'gamma': 0.1,
        'alpha': 0.25,
    },
    'cs': {
        'population_size': 30,
        'pa': 0.15,
        'alpha': 0.005,
    },
    'hill_climbing': {
        'max_neighbors': 20,
        'step_size': 0.05,
    },
    'simulated_annealing': {
        'initial_temp': 1000,
        'cooling_rate': 0.95,
        'min_temp': 1e-3,
    },
}

# PROBLEMS TO TEST
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