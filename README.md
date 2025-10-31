# Algorithm Comparison Project

So sánh hiệu quả giữa các thuật toán Swarm Intelligence và Traditional Search.

## 🚀 Quick Start

### 1. Cài đặt dependencies

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
- ✅ Chạy 10 lần mỗi experiment (có thể thay đổi)
- ✅ Lưu kết quả vào `results/`
- ✅ In summary table

### 3. Xem kết quả

Kết quả được lưu trong folder `results/` dạng JSON file:
```
results/results_20251031_143022.json
```

## 📊 Cấu hình

Chỉnh sửa tham số trong file `config.py`:

```python
EXPERIMENT_CONFIG = {
    'n_runs': 10,      # Số lần chạy (khuyến nghị: 30)
    'max_iter': 100,   # Số iteration
    'dimensions': [10], # Số chiều test
}
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
- BFS (optional)
- A* (optional)

## 📈 Bài toán Test

### Continuous Problems:
- Sphere Function (dễ, unimodal)
- Rastrigin Function (khó, multimodal)
- Rosenbrock Function (valley)
- Ackley Function (multimodal)
- Griewank Function
- Schwefel Function

## 📁 Cấu trúc Project

```
source code/
├── main.py                  # ⭐ File chính - Chạy file này!
├── config.py                # Cấu hình
├── requirements.txt         # Dependencies
│
├── algorithms/              # Các thuật toán
│   ├── base_optimizer.py
│   ├── swarm_algs/
│   │   ├── pso.py
│   │   ├── abc.py
│   │   └── ...
│   └── traditional_algs/
│       └── hill_climbing.py
│
├── problems/                # Các bài toán
│   ├── base_problem.py
│   ├── continuous_prob.py
│   └── discrete_prob.py
│
├── experiments/             # Experiment runners
│   ├── run_continuous_tests.py
│   └── run_discrete_tests.py
│
├── utils/                   # Utilities
│   ├── metrics.py
│   └── visualization_tools.py
│
└── results/                 # Kết quả (tự động tạo)
    └── results_xxx.json
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
