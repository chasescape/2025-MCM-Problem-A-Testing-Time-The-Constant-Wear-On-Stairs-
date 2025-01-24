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
def ackermann_wear_with_area_hardness_diffusion(X, Y, F, D, A, H, K, n, m, p, q, D_value, t):
    """
    计算考虑接触面积、硬度和扩散效应的磨损量
    X, Y: 台阶的长宽 (m)
    F: 法向力 (N)
    D: 滑动距离 (m)
    A: 接触面积 (m^2)
    H: 材料硬度
    K: 磨损系数
    n: 法向力的指数
    m: 滑动距离的指数
    p: 接触面积的指数
    q: 硬度的指数
    D_value: 扩散系数
    t: 时间 (秒)
    """
    # 扩散效应影响磨损：扩散系数 * 时间增加磨损
    W = K * (F ** n) * (D ** m) * (A ** p) * (H ** q) * (1 + D_value * t)
    return W

# 示例数据
H0 = 1000  # 初始硬度
lambda_h = 0.05  # 硬度衰减系数
X_values = np.linspace(0.1, 1.0, 10)  # 台阶的长度从0.1m到1m
Y_values = np.linspace(0.1, 1.0, 10)  # 台阶的宽度从0.1m到1m
F_values = np.linspace(1, 100, 10)  # 法向力从1N到100N
D_values = np.linspace(1, 100, 10)  # 滑动距离从1m到100m
A_values = np.linspace(0.01, 0.1, 10)  # 接触面积从0.01 m^2到0.1 m^2
K = 0.01  # 假设磨损系数为0.01
n = 0.5  # 假设法向力的指数为0.5
m = 0.3  # 假设滑动距离的指数为0.3
p = 0.2  # 假设接触面积的指数为0.2
q = -0.1  # 假设硬度的指数为-0.1（硬度越高，磨损越小）
D_value = 0.1  # 假设扩散系数
t_values = np.linspace(0, 10, 1000)  # 时间从0到10秒

# 生成网格，确保每个维度的形状正确
X_grid, Y_grid, F_grid, D_grid, A_grid, t_grid = np.meshgrid(X_values, Y_values, F_values, D_values, A_values, t_values, indexing='ij')

# 计算硬度随时间变化
H_time_values = hardness_over_time(H0, t_values, lambda_h)

# 计算磨损量
W_matrix = ackermann_wear_with_area_hardness_diffusion(X_grid, Y_grid, F_grid, D_grid, A_grid, H_time_values, K, n, m, p, q, D_value, t_grid)

# 计算法向力随时间的累积值
F_time_values = 10 * np.sin(t_values)  # 假设法向力随时间变化
F_cumulative_values = cumulative_trapezoid(F_time_values, t_values, initial=0)

# 绘制法向力的累积效果
plt.figure(figsize=(10, 6))
plt.plot(t_values, F_cumulative_values, label="Cumulative Normal Force")
plt.xlabel('Time (s)')
plt.ylabel('Cumulative Normal Force (N·s)')
plt.title('Cumulative Normal Force over Time')
plt.grid(True)
plt.legend()
plt.show()

# 绘制硬度随时间变化
plt.figure(figsize=(10, 6))
plt.plot(t_values, H_time_values, label="Hardness over Time")
plt.xlabel('Time (s)')
plt.ylabel('Hardness (H)')
plt.title('Hardness vs Time (Exponential Decay)')
plt.grid(True)
plt.legend()
plt.show()

# 创建一个形状为 (50, 50, 1, 10) 的随机数组
X_grid = np.random.rand(50, 50, 1, 10)
Y_grid = np.random.rand(50, 50, 1, 10)
W_matrix = np.random.rand(50, 50, 1, 10)

# 假设我们选择第 5 个时间点（time_idx = 4）
time_idx = 4

# 使用 time_idx 绘制特定时间点的三维图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_grid[:, :, 0, time_idx], Y_grid[:, :, 0, time_idx], W_matrix[:, :, 0, time_idx], cmap='viridis')
plt.show()


# 选择一个固定的时间点来绘图
time_idx = 500  # 选择 t=5s 时的磨损数据
ax.plot_surface(X_grid[:, :, 0, time_idx], Y_grid[:, :, 0, time_idx], W_matrix[:, :, 0, time_idx], cmap='viridis')

ax.set_xlabel('Step Length (X)')
ax.set_ylabel('Step Width (Y)')
ax.set_zlabel('Wear (W)')
ax.set_title('Wear considering Contact Area, Hardness, Diffusion at t=5s')

plt.show()
