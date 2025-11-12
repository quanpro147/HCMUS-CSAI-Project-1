# Algorithm Comparison Project

So sánh hiệu quả giữa các thuật toán Swarm Intelligence và Traditional Search.

## 🚀 How to run
### 1. Cài đặt môi trường bằng conda (Optinal)

```bash
conda create -n swarm_env python=3.10 -y

```
### 2. Kích hoạt môi trường

```bash
conda activate swarm_env

```
### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt

```

### 2. Chạy experiments

**Cách đơn giản nhất - Chạy tất cả:**

```bash
python main.py
```

Chương trình sẽ:
- ✅ Test tất cả thuật toán trên tất cả bài toán
- ✅ Lưu kết quả vào `results/`
- ✅ In summary table

### 3. Xem kết quả

Kết quả được lưu trong folder `results/` dạng JSON file:
```
results/continuous_results.json
results/discrete_results.json
```

## 🤖 Thuật toán

### Swarm Intelligence:
- PSO - Particle Swarm Optimization
- ABC - Artificial Bee Colony
- ACO - Ant Colony Optimization
- FA - Firefly Algorithm
- CS - Cuckoo Search

### Traditional Search:
- Hill Climbing
- Simulated annealing
- BFS
- A*

## 📈 Bài toán Test

### Continuous Problems:
- Sphere Function
- Rastrigin Function 
- Ackley Function 
### Discrete Problem:
- TSP
- GridPathFinding

## 📁 Cấu trúc Project

```
source code/
├── main.py                     # File chính - chạy toàn bộ experiment
├── config.py                   # File cấu hình
├── requirements.txt            # Danh sách dependencies
│
├── algorithms/                 # Thư mục chứa thuật toán
│   ├── base_optimizer.py
│   ├── swarm_algs/             # Thuật toán Swarm
│   │   ├── pso.py
│   │   ├── abc.py
│   │   ├── aco.py
│   │   ├── fa.py
│   │   └── cs.py
│   └── traditional_algs/       # Thuật toán truyền thống
│       ├── hill_climbing.py
│       ├── simulated_annealing.py
│       ├── bfs.py
│       └── astar.py
│
├── problems/                   # Các bài toán
│   ├── base_problem.py
│   ├── continuous_prob.py
│   └── discrete_prob.py
│
├── experiments/                # Chạy thí nghiệm
│   ├── run_continuous_tests.py
│   └── run_discrete_tests.py
│
├── testcases/                  # Bộ testcase
│   ├── continuous_testcases.json
│   └── discrete_testcases.json
│
├── results/                    # Kết quả đầu ra
│   ├── continuous_results.json
│   └── discrete_results.json
│
├── utils.py                    # Hàm tiện ích (tính metric)
├── testcases_loader.py         # Hàm load các test case
├── visualize.ipynb             # Notebook để vẽ và phân tích kết quả
└── README.md                   # Tài liệu hướng dẫn

```

## 🔬 Chạy từng phần (Advanced)

### Chỉ chạy continuous problems:
```bash
python experiments/run_continuous_tests.py
```

### Chỉ chạy discrete problems:
```bash
python experiments/run_discrete_tests.py
```

## ⚙️ Tuning Parameters

Để thay đổi tham số thuật toán, sửa trong `config.py`:

```python
ALGORITHM_PARAMS = {
    'pso': {
        'population_size': 30,
        'w': 0.7,
        'c1': 1.5,
        'c2': 1.5,
    },
    ...
}
```

## 📚 References

- PSO: Kennedy & Eberhart (1995)
- ABC: Karaboga (2005)
- FA: Yang (2008)
- CS: Yang & Deb (2009)
