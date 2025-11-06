import numpy as np
from typing import List, Tuple
from .base_problem import DiscreteProblem

class TravelingSalesmanProblem(DiscreteProblem):
    """
    Bài toán người du lịch (TSP).
    
    Mỗi nghiệm là một permutation của các thành phố (0 đến n_cities-1).
    Mục tiêu là tìm tour ngắn nhất đi qua tất cả các thành phố đúng một lần.
    """
    
    def __init__(self, name: str = "TSP", n_cities: int = 20, 
                 distance_matrix: np.ndarray = None, coords: np.ndarray = None, seed: int = None):
        """
        Args:
            name: Tên bài toán
            n_cities: Số lượng thành phố
            distance_matrix: Ma trận khoảng cách giữa các thành phố (tùy chọn)
            coords: Tọa độ các thành phố shape (n_cities, 2) (tùy chọn)
            seed: Seed cho random
        """
        super().__init__(name)
        self.n_cities = n_cities
        self.coords = None
        
        # Ưu tiên: coords > distance_matrix > generate random
        if coords is not None:
            # Tạo distance matrix từ coords
            self.coords = np.array(coords)
            if self.coords.shape[0] != n_cities:
                raise ValueError(f"coords phải có {n_cities} thành phố, nhưng có {self.coords.shape[0]}")
            self.distance_matrix = self._compute_distance_matrix_from_coords(self.coords)
        elif distance_matrix is not None:
            # Dùng distance matrix có sẵn
            if distance_matrix.shape != (n_cities, n_cities):
                raise ValueError("Distance matrix phải có kích thước (n_cities, n_cities)")
            self.distance_matrix = distance_matrix
        else:
            # Generate ngẫu nhiên
            self.distance_matrix = self._generate_distance_matrix(seed)
        
        self.optimal_value = None
    
    def _compute_distance_matrix_from_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Tính ma trận khoảng cách từ tọa độ các thành phố.
        
        Args:
            coords: Ma trận tọa độ shape (n_cities, 2)
            
        Returns:
            np.ndarray: Ma trận khoảng cách Euclidean
        """
        n = len(coords)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(coords[i] - coords[j])
                distances[i, j] = dist
                distances[j, i] = dist
        return distances

    def _generate_distance_matrix(self, seed: int = None) -> np.ndarray:
        """
        Tạo ma trận khoảng cách ngẫu nhiên.
        
        Args:
            seed: Random seed
            
        Returns:
            np.ndarray: Ma trận khoảng cách
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Tạo tọa độ ngẫu nhiên cho các thành phố
        self.coords = np.random.rand(self.n_cities, 2) * 100
        
        # Tính khoảng cách Euclidean
        distances = np.zeros((self.n_cities, self.n_cities))
        for i in range(self.n_cities):
            for j in range(i + 1, self.n_cities):
                dist = np.linalg.norm(self.coords[i] - self.coords[j])
                distances[i, j] = dist
                distances[j, i] = dist
        return distances
    
    def evaluate(self, solution: np.ndarray) -> float:
        """
        Tính độ dài của tour.
        
        Args:
            solution: Một permutation của các thành phố
            
        Returns:
            float: Tổng độ dài của tour
        """
        total_distance = 0
        for i in range(len(solution)):
            from_city = solution[i]
            to_city = solution[(i + 1) % len(solution)]
            total_distance += self.distance_matrix[from_city, to_city]
        return total_distance

    def random_solution(self) -> np.ndarray:
        """
        Tạo một tour ngẫu nhiên.
        
        Returns:
            np.ndarray: Một permutation ngẫu nhiên của các thành phố
        """
        return np.random.permutation(self.n_cities)
    
    def is_valid(self, solution: np.ndarray) -> bool:
        """
        Kiểm tra tính hợp lệ của tour.
        
        Tour hợp lệ phải:
        - Có độ dài bằng số thành phố
        - Chứa mỗi thành phố đúng một lần
        - Các chỉ số thành phố phải nằm trong khoảng hợp lệ
        
        Args:
            solution: Tour cần kiểm tra
            
        Returns:
            bool: True nếu tour hợp lệ, False nếu không
        """
        solution = np.array(solution)
        return (len(solution) == self.n_cities and
                len(set(solution)) == self.n_cities and
                all(0 <= city < self.n_cities for city in solution))
    
    def swap_cities(self, solution: np.ndarray, i: int, j: int) -> np.ndarray:
        """
        Hoán đổi vị trí hai thành phố trong tour.
        
        Args:
            solution: Tour hiện tại
            i, j: Vị trí hai thành phố cần hoán đổi
            
        Returns:
            np.ndarray: Tour mới sau khi hoán đổi
        """
        new_solution = solution.copy()
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
        return new_solution
    
    def two_opt(self, solution: np.ndarray, i: int, j: int) -> np.ndarray:
        """
        Áp dụng phép biến đổi 2-opt.
        
        Args:
            solution: Tour hiện tại
            i, j: Đoạn cần đảo ngược
            
        Returns:
            np.ndarray: Tour mới sau khi áp dụng 2-opt
        """
        new_solution = solution.copy()
        new_solution[i:j+1] = new_solution[i:j+1][::-1]
        return new_solution
        
    def get_neighbors(self, solution: np.ndarray, method: str = 'swap') -> List[np.ndarray]:
        """
        Sinh các neighbors của một nghiệm.
        
        Args:
            solution: Nghiệm hiện tại
            method: Phương pháp sinh neighbor ('swap' hoặc '2-opt')
            
        Returns:
            List[np.ndarray]: Danh sách các neighbors
        """
        neighbors = []
        n = len(solution)
        
        if method == 'swap':
            for i in range(n):
                for j in range(i+1, n):
                    neighbors.append(self.swap_cities(solution, i, j))
        elif method == '2-opt':
            for i in range(n-1):
                for j in range(i+2, n):
                    neighbors.append(self.two_opt(solution, i, j))
        else:
            raise ValueError("method phải là 'swap' hoặc '2-opt'")
            
        return neighbors
    

class GridPathfindingProblem(DiscreteProblem):
    """
    Bài toán tìm đường đi trên lưới (Grid) cho A*.
    
    Quy ước:
    - 0: Ô trống (có thể đi)
    - 1: Tường (không thể đi)
    """
    
    def __init__(self, grid: np.ndarray = None, start: Tuple[int, int] = None, 
                 goal: Tuple[int, int] = None, height: int = 10, width: int = 10,
                 obstacle_ratio: float = 0.2, seed: int = None,
                 name: str = "Grid Pathfinding"):
        """
        Args:
            grid: Ma trận 2D numpy (0 là đường, 1 là tường). Nếu None, tự generate.
            start: Tọa độ (y, x) của điểm bắt đầu. Nếu None, dùng (0, 0).
            goal: Tọa độ (y, x) của điểm kết thúc. Nếu None, dùng (height-1, width-1).
            height: Chiều cao lưới (chỉ dùng khi grid=None).
            width: Chiều rộng lưới (chỉ dùng khi grid=None).
            obstacle_ratio: Tỷ lệ chướng ngại vật (0.0-1.0, chỉ dùng khi grid=None).
            seed: Random seed.
            name: Tên bài toán.
        """
        super().__init__(name)
        
        # Tạo hoặc load grid
        if grid is not None:
            self.grid = np.array(grid)
            self.height, self.width = self.grid.shape
        else:
            # Generate random grid
            self.height = height
            self.width = width
            self.grid = self._generate_random_grid(obstacle_ratio, seed)
        
        # Set start và goal
        self.start_pos = start if start is not None else (0, 0)
        self.goal_pos = goal if goal is not None else (self.height - 1, self.width - 1)
        
        # Validate
        if not (0 <= self.start_pos[0] < self.height and 0 <= self.start_pos[1] < self.width):
            raise ValueError(f"Điểm bắt đầu {self.start_pos} nằm ngoài lưới {self.height}x{self.width}.")
        if not (0 <= self.goal_pos[0] < self.height and 0 <= self.goal_pos[1] < self.width):
            raise ValueError(f"Điểm kết thúc {self.goal_pos} nằm ngoài lưới {self.height}x{self.width}.")
        if self.grid[self.start_pos] == 1:
            # Nếu start là tường, xóa tường đó
            self.grid[self.start_pos] = 0
        if self.grid[self.goal_pos] == 1:
            # Nếu goal là tường, xóa tường đó
            self.grid[self.goal_pos] = 0
    
    def _generate_random_grid(self, obstacle_ratio: float = 0.2, seed: int = None) -> np.ndarray:
        """
        Tạo lưới ngẫu nhiên với chướng ngại vật.
        
        Args:
            obstacle_ratio: Tỷ lệ ô là tường (0.0-1.0).
            seed: Random seed.
            
        Returns:
            np.ndarray: Grid với 0 (đường) và 1 (tường).
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Tạo grid ngẫu nhiên
        grid = (np.random.rand(self.height, self.width) < obstacle_ratio).astype(int)
        
        # Đảm bảo start và goal không bị tường
        grid[0, 0] = 0  # start
        grid[self.height - 1, self.width - 1] = 0  # goal
        
        return grid

    def get_start_state(self) -> Tuple[int, int]:
        """(Hàm này A* cần) Trả về trạng thái bắt đầu."""
        return self.start_pos

    def is_goal_state(self, state: Tuple[int, int]) -> bool:
        """(Hàm này A* cần) Kiểm tra state có phải là đích."""
        return state == self.goal_pos

    def get_neighbors(self, state: Tuple[int, int]) -> List[Tuple[Tuple[int, int], int]]:
        """
        (Hàm này A* cần) Lấy các hàng xóm hợp lệ của một state.
        
        Returns:
            List[ (neighbor_state, cost) ]
        """
        y, x = state
        neighbors = []
        
        # Duyệt 4 hướng (trên, dưới, trái, phải)
        for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ny, nx = y + dy, x + dx
            
            # 1. Kiểm tra có nằm trong biên (bounds)
            if 0 <= ny < self.height and 0 <= nx < self.width:
                # 2. Kiểm tra có phải là tường (obstacle)
                if self.grid[ny, nx] == 0:
                    cost = 1 # Giả sử chi phí di chuyển là 1
                    neighbors.append( ((ny, nx), cost) )
                    
        return neighbors

    def get_heuristic(self, state: Tuple[int, int]) -> float:
        """
        (Hàm này A* cần) Tính heuristic (h) (khoảng cách Manhattan).
        """
        return abs(state[0] - self.goal_pos[0]) + abs(state[1] - self.goal_pos[1])

    def evaluate(self, solution):
        """A* không dùng hàm này trực tiếp."""
        if not isinstance(solution, list):
            return float('inf')
        return len(solution) - 1

    def random_solution(self):
        """Không áp dụng cho A*."""
        return None

    def is_valid(self, solution):
        """Kiểm tra đường đi có hợp lệ (không đi vào tường)."""
        if not isinstance(solution, list):
            return False
        for (y, x) in solution:
            if not (0 <= y < self.height and 0 <= x < self.width) or self.grid[y, x] == 1:
                return False
        return True
    
