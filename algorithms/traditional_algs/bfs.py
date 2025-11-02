from ..base_optimizer import TraditionalOptimizer
from collections import deque
from typing import Any, Tuple

class BFS(TraditionalOptimizer):
    """
    Breadth-First Search (BFS) Algorithm implementation.
    """

    def __init__(self, name: str = "Breadth-First Search"):
        """
        Args:
            name: Tên của thuật toán
        """
        super().__init__(name)

    def optimize(self, problem, max_iter: int = 1000, **kwargs) -> Tuple[Any, float]:
        """
        Thực hiện tìm kiếm theo chiều rộng (BFS).
        
        Args:
            problem: Bài toán cần tối ưu, có method:
                - random_solution() hoặc initial_state()
                - get_neighbors(state)
                - evaluate(state)
                - is_goal(state) (tùy chọn)
            max_iter: Số node tối đa được mở rộng
            **kwargs: Tham số bổ sung
            
        Returns:
            Tuple[state, fitness]: Trạng thái tốt nhất và giá trị fitness tương ứng
        """

        # Lấy trạng thái khởi tạo
        if hasattr(problem, "initial_state"):
            start = problem.initial_state()
        else:
            start = problem.random_solution()

        start_fitness = problem.evaluate(start)
        self.function_evaluations += 1

        self.best_solution = start
        self.best_fitness = start_fitness
        self._update_convergence()

        # Hàng đợi cho BFS
        frontier = deque([start])
        visited = set()
        visited.add(tuple(start))  # giả định state có thể hash được

        self.search_path.append(start.copy())

        iterations = 0

        while frontier and iterations < max_iter:
            iterations += 1
            self.iterations_done += 1

            current = frontier.popleft()
            current_fitness = problem.evaluate(current)
            self.function_evaluations += 1

            # Cập nhật nghiệm tốt nhất
            if current_fitness < self.best_fitness:
                self.best_solution = current.copy()
                self.best_fitness = current_fitness

            # Kiểm tra goal nếu có
            if hasattr(problem, "is_goal") and problem.is_goal(current):
                break

            # Sinh neighbor
            neighbors = problem.get_neighbors(current)

            for neighbor in neighbors:
                key = tuple(neighbor)
                if key not in visited:
                    visited.add(key)
                    frontier.append(neighbor)
                    self.search_path.append(neighbor.copy())

            self._update_convergence()

        return self.best_solution, self.best_fitness
