import numpy as np

# --- Common Metrics ---
def compute_basic_stats(values):
    values = np.array(values)
    stats = {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'median': float(np.median(values)),
        'q25': float(np.percentile(values, 25)),
        'q75': float(np.percentile(values, 75)),
        'all_values': [float(v) for v in values]
    }
    return stats

def compute_error_from_optimal(values, optimal):
    values = np.array(values)
    return {
        'mean': float(np.mean(values) - optimal),
        'min': float(np.min(values) - optimal)
    }

# --- For discrete summary output (mean/std only) ---
def compute_summary_metrics(values):
    values = np.array(values)
    return {'mean': float(np.mean(values)), 'std': float(np.std(values))}

# --- Convergence Speed ---
def compute_convergence_speed(convergence_curve, target_factor=0.1):
    """
    Tính số iteration cần để giảm tới target_factor*initial (mặc định 10% giá trị ban đầu)
    Return: số iteration (int), càng nhỏ càng nhanh.
    """
    if len(convergence_curve) == 0:
        return None
    initial = convergence_curve[0]
    target = initial * target_factor
    for i, val in enumerate(convergence_curve):
        if val <= target:
            return i
    return len(convergence_curve)

# --- Computational Complexity ---
def compute_time_complexity(time_values):
    """
    Time complexity: trả về dict stats về thời gian chạy (mean, std, min, max...)
    """
    return compute_basic_stats(time_values)

def compute_space_complexity(memory_usages):
    """
    Space complexity: thống kê peak memory used, input là list memory sử dụng cho mỗi run.
    """
    return compute_basic_stats(memory_usages)

# --- Robustness ---
def compute_robustness_metrics(performance_values):
    """
    Robustness = độ ổn định performance qua nhiều lần chạy (mean, std càng nhỏ càng tốt)
    """
    return compute_basic_stats(performance_values)

# --- Scalability ---
def compute_scalability_metric(performance_vs_size: dict):
    """
    Đo mức độ thay đổi performance khi tăng kích thước/problem size.
    Input: dict {size: performance}
    Output: dict với hệ số regress (bậc 1), list (size, perf)
    """
    sizes = list(performance_vs_size.keys())
    values = list(performance_vs_size.values())
    coeffs = np.polyfit(sizes, values, deg=1)
    return {
        'sizes': sizes,
        'performances': values,
        'linear_coef': coeffs[0],
        'intercept': coeffs[1]
    }
