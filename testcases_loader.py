import json
import os
from typing import List, Dict, Any

from problems import (
    SphereFunction, 
    RastriginFunction, 
    AckleyFunction, 
    TravelingSalesmanProblem, 
    GridPathfindingProblem,
)

continuous_registry = {
    "sphere": SphereFunction,
    "rastrigin": RastriginFunction,
    "ackley": AckleyFunction,
}

discrete_registry = {
    "tsp": TravelingSalesmanProblem,
    "gridpathfinding": GridPathfindingProblem,
}


def create_problem(name: str, prob_type: str, **kwargs):
    """
    Khởi tạo đối tượng bài toán dựa trên tên và loại.

    Args:
        name (str): Tên bài toán (Sphere, TSP, GridPathfinding, ...)
        prob_type (str): Loại ("continuous" hoặc "discrete")
        kwargs: Tham số khởi tạo (dim, bound, coords, grid, ...)

    Returns:
        Instance của bài toán tương ứng
    """
    name_lower = name.lower()
    prob_type = prob_type.lower()

    if prob_type == "continuous":
        if name_lower not in continuous_registry:
            raise ValueError(f" Unknown continuous problem '{name}'")
        return continuous_registry[name_lower](**kwargs)

    elif prob_type == "discrete":
        if name_lower not in discrete_registry:
            raise ValueError(f" Unknown discrete problem '{name}'")
        return discrete_registry[name_lower](**kwargs)

    else:
        raise ValueError(f" Unknown problem type '{prob_type}' (must be 'continuous' or 'discrete')")

def load_testcases(filepath: str) -> List[Any]:
    """
    Load test cases từ file JSON bất kỳ (continuous hoặc discrete).

    Args:
        filepath (str): Đường dẫn file JSON

    Returns:
        List[Any]: Danh sách đối tượng bài toán đã khởi tạo
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f" File not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    problems = []

    for name, info in data.items():
        prob_type = info.get("type", "continuous")

        if prob_type == "continuous":
            for test in info.get("tests", []):
                dim = test["dim"]
                bound = tuple(test["bound"])
                shift = test.get("shift", False)
                rotate = test.get("rotate", False)
                seed = test.get("seed", None)

                problem = create_problem(
                    name, prob_type,
                    dim=dim, bounds=bound, shift=shift, rotate=rotate, seed=seed
                )

                problem.test_id = test.get("id", f"{name}_{dim}D")
                problem.optimal_value = test.get("optimal_value", 0.0)
                problem.optimal_position = test.get("optimal_position", "unknown")

                problems.append(problem)

        elif prob_type == "discrete":
            # TSP
            if name.lower() == "tsp":
                for test in info.get("tests", []):
                    n_cities = test["n_cities"]
                    coords = test.get("coords", None)
                    distance_matrix = test.get("distance_matrix", None)
                    seed = test.get("seed", None)

                    problem = create_problem(
                        "tsp", "discrete",
                        n_cities=n_cities,
                        coords=coords,
                        distance_matrix=distance_matrix,
                        seed=seed
                    )
                    problem.test_id = test.get("id", f"TSP_{n_cities}cities")
                    problems.append(problem)

            # Grid Pathfinding
            elif name.lower() == "gridpathfinding":
                for test in info.get("tests", []):
                    grid = test["grid"]
                    start = tuple(test["start"])
                    goal = tuple(test["goal"])
                    problem = create_problem(
                        "gridpathfinding", "discrete",
                        grid=grid, start=start, goal=goal
                    )
                    problem.test_id = test.get("id", f"Grid_{len(grid)}x{len(grid[0])}")
                    problems.append(problem)

    print(f"Loaded {len(problems)} test cases từ '{os.path.basename(filepath)}'")
    return problems