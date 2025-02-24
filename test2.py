import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')  # 或者使用 'Agg' 后端
# 参数设置
k0 = 1.0  # 基准磨损系数
alpha = 0.1  # 温度对磨损系数的影响系数
beta = 0.05  # 湿度对磨损系数的影响系数
n = 2  # 表面粗糙度的指数
T_values = np.linspace(0, 100, 100)  # 温度范围，从 0 到 100 摄氏度
eta_values = np.linspace(0, 1, 100)  # 湿度范围，从 0 到 1
R_values = np.linspace(0.1, 5, 100)  # 粗糙度范围，从 0.1 到 5


# 计算磨损系数 k(T, eta, R)
def calculate_wear_coefficient(T, eta, R):
    f_T = np.exp(-alpha * T)  # 温度影响修正函数
    f_eta = 1 + beta * eta  # 湿度影响修正函数
    f_R = R ** n  # 粗糙度影响修正函数
    return k0 * f_T * f_eta * f_R


# 创建温度-湿度与磨损系数的关系图
K_T_eta_values = np.array([[calculate_wear_coefficient(T, eta, 1) for eta in eta_values] for T in T_values])

# 绘制温度-湿度对磨损系数的影响
plt.figure(figsize=(10, 6))
plt.contourf(eta_values, T_values, K_T_eta_values, 20, cmap='inferno')
plt.colorbar(label='Wear Coefficient (k)')
plt.xlabel('Humidity')
plt.ylabel('Temperature (°C)')
plt.title('Wear Coefficient vs Temperature and Humidity')
plt.grid(True)
plt.show()

# 创建粗糙度-温度与磨损系数的关系图
K_R_T_values = np.array([[calculate_wear_coefficient(T, 0.5, R) for R in R_values] for T in T_values])

# 绘制粗糙度-温度对磨损系数的影响
plt.figure(figsize=(10, 6))
plt.contourf(R_values, T_values, K_R_T_values, 20, cmap='viridis')
plt.colorbar(label='Wear Coefficient (k)')
plt.xlabel('Surface Roughness (R)')
plt.ylabel('Temperature (°C)')
plt.title('Wear Coefficient vs Surface Roughness and Temperature')
plt.grid(True)
plt.show()
