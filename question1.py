import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
import matplotlib

matplotlib.use('TkAgg')  # 或者使用 'Agg' 后端


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
    W = K * (F ** n) * d * (A ** p) * (H ** q) * (1 + D * t)
    return W


# 示例数据
H0 = 1000  # 初始硬度
lambda_h = 0.05  # 硬度衰减系数
F_values = np.linspace(1, 100, 10)  # 法向力从1N到100N
d_values = np.linspace(1, 100, 10)  # 滑动距离从1m到100m
A_values = np.linspace(0.01, 0.1, 10)  # 接触面积从0.01 m^2到0.1 m^2
K = 0.01  # 假设磨损系数为0.01
n = 0.5  # 假设法向力的指数为0.5
p = 0.2  # 假设接触面积的指数为0.2
q = -0.1  # 假设硬度的指数为-0.1（硬度越高，磨损越小）
D = 0.1  # 假设扩散系数
# 参数设置
m = 1.0  # 物体质量 (kg)
g = 9.81  # 重力加速度 (m/s^2)
omega = 2 * np.pi  # 角频率 (假设单位为 rad/s)
t_values = np.linspace(0, 10, 1000)  # 时间数组，范围 0 到 10 秒

# 生成网格，确保每个维度的形状正确
F_grid, D_grid, A_grid, H_grid = np.meshgrid(F_values, d_values, A_values, t_values, indexing='ij')

# 计算硬度随时间变化
H_time_values = hardness_over_time(H0, t_values, lambda_h)

# 计算磨损量
W_matrix = ackermann_wear_with_area_hardness_diffusion(F_grid, D_grid, A_grid, H_time_values, K, n, m, p, q, D,
                                                       t_values)

# 计算法向力随时间的累积值


# 计算法向力 Fn(t)
F_n_values = m * g * (1 + np.sin(omega * t_values))

# 计算法向力的累积值
F_cumulative_values = cumulative_trapezoid(F_n_values, t_values, initial=0)

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

# 绘制磨损随法向力和接触面积变化的效果
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 选择一个固定的时间点来绘图
time_idx = 500  # 选择 t=5s 时的磨损数据
ax.plot_surface(F_grid[:, :, 0, time_idx], D_grid[:, :, 0, time_idx], W_matrix[:, :, 0, time_idx], cmap='viridis')

ax.set_xlabel('Normal Force (F)')
ax.set_ylabel('Sliding Distance (d)')
ax.set_zlabel('Wear (W)')
ax.set_title('Wear considering Contact Area, Hardness, and Diffusion at t=5s')

plt.show()
