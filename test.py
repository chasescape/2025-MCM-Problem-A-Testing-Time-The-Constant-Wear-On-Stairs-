import matplotlib
matplotlib.use('TkAgg')  # 或者使用 'Agg' 后端

import numpy as np
import matplotlib.pyplot as plt

# 定义考虑接触面积和硬度的磨损定理函数
def ackermann_wear_with_area_and_hardness(F, d, A, H, K, n, m, p, q):
    """
    计算考虑接触面积和硬度的磨损量
    F: 法向力 (N)
    d: 相对滑动距离 (m)
    A: 接触面积 (m^2)
    H: 材料硬度
    K: 磨损系数
    n: 法向力指数
    m: 滑动距离指数
    p: 接触面积指数
    q: 硬度指数
    """
    W = K * (F ** n) * (d ** m) * (A ** p) * (H ** q)
    return W

# 示例数据
F_values = np.linspace(1, 100, 10)  # 法向力从1N到100N
d_values = np.linspace(1, 100, 10)  # 滑动距离从1m到100m
A_values = np.linspace(0.01, 0.1, 10)  # 接触面积从0.01 m^2到0.1 m^2
H_values = np.linspace(100, 1000, 10)  # 材料硬度从100到1000（假设硬度单位）
K = 0.01  # 假设磨损系数为0.01
n = 0.5   # 假设法向力的指数为0.5
m = 0.3   # 假设滑动距离的指数为0.3
p = 0.2   # 假设接触面积的指数为0.2
q = -0.1  # 假设硬度的指数为-0.1（硬度越高，磨损越小）

# 生成网格，确保每个维度的形状正确
F_grid, D_grid, A_grid, H_grid = np.meshgrid(F_values, d_values, A_values, H_values, indexing='ij')

# 计算磨损量
W_matrix = ackermann_wear_with_area_and_hardness(F_grid, D_grid, A_grid, H_grid, K, n, m, p, q)

# 绘制3D图形来表示磨损量随法向力、滑动距离、接触面积和硬度的变化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 选择适当的维度来绘图，比如选择一个固定的滑动距离和接触面积
ax.plot_surface(F_grid[:, :, 0, 0], D_grid[:, :, 0, 0], W_matrix[:, :, 0, 0], cmap='viridis')

# 设置标签
ax.set_xlabel('Normal Force (F)')
ax.set_ylabel('Sliding Distance (d)')
ax.set_zlabel('Wear (W)')
ax.set_title('Wear considering Contact Area and Hardness')

plt.show()
