import matplotlib
matplotlib.use('TkAgg')  # 或者使用 'Agg' 后端

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# 定义硬度随时间变化的函数（指数衰减）
def hardness_over_time(H0, t, lambda_h):
    """
    计算随时间变化的硬度
    H0: 初始硬度
    t: 时间 (秒)
    lambda_h: 硬度衰减系数
    """
    return H0 * np.exp(-lambda_h * t)

# 定义考虑接触面积、硬度和扩散效应的磨损量函数
def ackermann_wear_with_area_hardness_diffusion(F, d, A, H, K, n, m, p, q, D, t):
    """
    计算考虑接触面积、硬度和扩散效应的磨损量
    F: 法向力 (N)
    d: 相对滑动距离 (m)
    A: 接触面积 (m^2)
    H: 材料硬度
    K: 磨损系数
    n: 法向力的指数
    m: 滑动距离的指数
    p: 接触面积的指数
    q: 硬度的指数
    D: 扩散系数
    t: 时间 (秒)
    """
    # 考虑扩散效应的磨损量：扩散效应会改变磨损的分布
    W = K * (F ** n) * (d ** m) * (A ** p) * (H ** q) * (1 + D * t)
    return W

# 定义一个高斯分布的磨损函数，用于模拟磨损在台阶中间的分布
def gaussian_wear_distribution(L, W, F, d, A, H, K, n, m, p, q, D, t, sigma=0.1):
    """
    计算磨损在台阶长宽区域的分布，磨损集中在中间部分
    L: 台阶长度
    W: 台阶宽度
    F: 法向力 (N)
    d: 滑动距离 (m)
    A: 接触面积 (m^2)
    H: 硬度
    K: 磨损系数
    n: 法向力的指数
    m: 滑动距离的指数
    p: 接触面积的指数
    q: 硬度的指数
    D: 扩散系数
    t: 时间 (秒)
    sigma: 高斯分布的标准差，用于控制磨损集中度
    """
    # 创建台阶长宽网格
    x = np.linspace(-L / 2, L / 2, 100)  # X 方向
    y = np.linspace(-W / 2, W / 2, 100)  # Y 方向
    X, Y = np.meshgrid(x, y)

    # 计算每个点的磨损，使用高斯分布模拟磨损集中在中心
    gaussian_distribution = np.exp(-((X**2 + Y**2) / (2 * sigma**2)))

    # 计算硬度随时间变化
    H_time_values = hardness_over_time(H, t, lambda_h=0.05)

    # 计算磨损量
    W_matrix = ackermann_wear_with_area_hardness_diffusion(F, d, A, H_time_values, K, n, m, p, q, D, t)

    # 将磨损分布与高斯分布结合，得到实际磨损量
    W_total = W_matrix * gaussian_distribution

    return X, Y, W_total

# 示例数据
H0 = 1000  # 初始硬度
F = 50  # 假设法向力
d = 0.05  # 假设滑动距离
A = 0.02  # 假设接触面积
K = 0.01  # 磨损系数
n = 0.5  # 法向力的指数
m = 0.3  # 滑动距离的指数
p = 0.2  # 接触面积的指数
q = -0.1  # 硬度的指数
D = 0.1  # 扩散系数
t = 5  # 时间
L = 2  # 台阶长度
W = 1  # 台阶宽度

# 计算磨损分布
X, Y, W_total = gaussian_wear_distribution(L, W, F, d, A, H0, K, n, m, p, q, D, t)

# 绘制磨损分布的热力图
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, W_total, 50, cmap='viridis')
plt.colorbar(label='Wear (W)')
plt.xlabel('Step Length (L)')
plt.ylabel('Step Width (W)')
plt.title('Wear Distribution on Step Surface')
plt.show()
