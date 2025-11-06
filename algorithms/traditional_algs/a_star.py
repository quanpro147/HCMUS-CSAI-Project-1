# Kiên
from algorithms.base_optimizer import TraditionalOptimizer
import numpy as np
import heapq  # Dùng Hàng đợi ưu tiên (Priority Queue)
from problems.discrete_prob import DiscreteProblem # Import rõ ràng để type hint

# ==============================================================================
# Lớp Helper cho A* (Phải nằm ngoài class AStar)
# ==============================================================================
class AStarNode:
    """Lớp lưu trữ thông tin của một nút trên đồ thị/lưới."""
    def __init__(self, f: float, h: float, state: tuple, parent: 'AStarNode' = None, g: float = 0):
        self.f = f  # f = g + h
        self.h = h  # Chi phí heuristic
        self.g = g  # Chi phí thực tế
        self.state = state # Trạng thái (ví dụ: tuple (y, x))
        self.parent = parent
        
    # So sánh các nút (heapq là min-heap nên sẽ ưu tiên f nhỏ nhất)
    def __lt__(self, other: 'AStarNode') -> bool:
        if self.f == other.f:
            # Nếu f bằng nhau, ưu tiên h nhỏ hơn (để phá vỡ sự cân bằng)
            return self.h < other.h
        return self.f < other.f
        
    # Dùng để so sánh state (cần hashable)
    def __eq__(self, other: object) -> bool:
        if isinstance(other, AStarNode):
            return self.state == other.state
        return False
        
    # Dùng cho set()
    def __hash__(self) -> int:
        return hash(self.state)

# ==============================================================================
# Lớp thuật toán A*
# ==============================================================================
class AStar(TraditionalOptimizer):
    """
    Triển khai thuật toán tìm kiếm A*.
    Kế thừa từ TraditionalOptimizer.
    """
    
    def __init__(self, **kwargs):
        """
        Khởi tạo thuật toán A*.
        """
        super().__init__(name="A* Search")

    def optimize(self, problem: 'DiscreteProblem', max_iter: int = 10000, **kwargs) -> tuple:
        """
        Hàm tìm kiếm chính, được gọi bởi self.run().
        
        Args:
            problem: Đối tượng bài toán (phải là DiscreteProblem và 
                     cung cấp các hàm: get_start_state, is_goal_state, 
                     get_neighbors, get_heuristic).
            max_iter: Số nút tối đa được mở rộng (để tránh lặp vô hạn).
            
        Returns:
            Tuple (best_solution, best_fitness).
            best_solution là list các state (đường đi).
            best_fitness là chi phí (g-score) của đường đi.
        """
        
        # Cấu trúc dữ liệu của A*
        open_set = []  # Hàng đợi ưu tiên (min-heap)
        closed_set = set() # Các nút đã duyệt (dùng set<tuple> để truy cập O(1))
        
        # Lưu trữ chi phí
        # Dùng dict để truy cập O(1), key là state (tuple), value là g_score
        g_scores = {} 
        
        # 1. Khởi tạo
        try:
            start_state = problem.get_start_state()
            g_scores[start_state] = 0
            
            h_start = problem.get_heuristic(start_state)
            self.function_evaluations += 1 # Đếm 1 lần gọi heuristic
            
            f_start = 0 + h_start
            start_node = AStarNode(f=f_start, h=h_start, g=0, state=start_state)
            
            heapq.heappush(open_set, start_node)
            
        except AttributeError as e:
            print(f"Lỗi: Đối tượng 'problem' thiếu phương thức cần thiết cho A*. {e}")
            print("Bài toán của bạn (ví dụ: GridPathfindingProblem) CẦN có 4 hàm: ")
            print("get_start_state(), is_goal_state(state), get_neighbors(state), get_heuristic(state)")
            return None, float('inf')

        # 2. Vòng lặp tìm kiếm
        while open_set and self.iterations_done < max_iter:
            
            # Lấy nút có f-score nhỏ nhất từ open_set
            current_node = heapq.heappop(open_set)
            
            # Nếu đã có trong closed_set, bỏ qua
            if current_node.state in closed_set:
                continue
                
            closed_set.add(current_node.state)

            # Cập nhật metrics
            self.iterations_done += 1 # Đếm 1 lần mở rộng nút
            self.search_path.append(current_node.state) # Lưu lại đường đi tìm kiếm

            # 3. Kiểm tra Đích
            if problem.is_goal_state(current_node.state):
                # TÌM THẤY ĐÍCH! Tái tạo đường đi
                path = self._reconstruct_path(current_node)
                self.best_solution = path
                self.best_fitness = current_node.g # Fitness là chi phí đường đi
                
                # A* không hội tụ, nó tìm thấy 1 lần
                self.convergence_curve = [self.best_fitness] 
                return self.best_solution, self.best_fitness

            # 4. Mở rộng các nút hàng xóm
            self.function_evaluations += 1 # Đếm 1 lần gọi get_neighbors
            for neighbor_state, cost in problem.get_neighbors(current_node.state):
                
                if neighbor_state in closed_set:
                    continue
                    
                tentative_g = current_node.g + cost
                
                # Kiểm tra xem đường đi mới này có tốt hơn đường cũ không
                if tentative_g < g_scores.get(neighbor_state, float('inf')):
                    # Đây là đường đi tốt hơn! Ghi lại
                    g_scores[neighbor_state] = tentative_g
                    
                    h = problem.get_heuristic(neighbor_state)
                    self.function_evaluations += 1 # Đếm 1 lần gọi heuristic
                    
                    f = tentative_g + h
                    
                    neighbor_node = AStarNode(f=f, h=h, g=tentative_g, 
                                            state=neighbor_state, parent=current_node)
                    
                    heapq.heappush(open_set, neighbor_node)

        # 5. Không tìm thấy đường đi (hết open_set hoặc quá max_iter)
        return None, float('inf')

    def _reconstruct_path(self, goal_node: AStarNode) -> list:
        """
        Helper: Tái tạo đường đi từ nút đích ngược về nút bắt đầu.
        """
        path = []
        current = goal_node
        while current is not None:
            path.append(current.state)
            current = current.parent
        return path[::-1] # Đảo ngược list