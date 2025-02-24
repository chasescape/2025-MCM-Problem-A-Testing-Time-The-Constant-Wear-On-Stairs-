import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')  # 或者使用 'Agg' 后端
# 参数设置
m = 60.0  # 物体质量 (kg)
g = 9.81  # 重力加速度 (m/s^2)
omega = 2 * np.pi  # 角频率 (rad/s)
v0 = 0.5  # 初始滑动速度 (m/s)
H0 = 1e9  # 初始硬度 (Pa)
A0 = 0.02  # 初始接触面积 (m²)
alpha = 0.1  # 温度对磨损系数的影响系数
beta = 0.05  # 湿度对磨损系数的影响系数
n = 2  # 表面粗糙度的指数
delta = 0.01  # 硬度衰减系数 (1/s)
gamma = 0.01  # 接触面积变化系数 (1/s)
k0 = 1.0  # 基准磨损系数
T0 = 25  # 初始温度 (°C)
eta0 = 50  # 初始湿度 (%)
R0 = 0.2  # 初始粗糙度 (m)
t_values = np.linspace(0, 10, 1000)  # 时间数组，范围 0 到 10 秒


# 环境因素影响磨损系数
def k(t, T, eta, R):
    return k0 * np.exp(-alpha * T) * (1 + beta * eta) * R ** n


# 法向力随时间变化
def Fn(t):
    return m * g * (1 + np.sin(omega * t))


# 滑动速度随时间变化
def v(t):
    return v0 * np.cos(omega * t)


# 硬度随时间变化
def H(t):
    return H0 * np.exp(-delta * t)


# 接触面积随时间变化
def A(t):
    return A0 * (1 + gamma * t)


# 自适应时间步长计算
def adaptive_time_step(integrand, t_max, initial_step=0.01, tolerance=1e-6):
    t = 0
    W = 0
    dt = initial_step
    while t < t_max:
        error_estimate = integrand(t) * dt  # 估算当前误差
        if error_estimate > tolerance:
            dt *= 0.9  # 如果误差过大，减小时间步长
        else:
            W += integrand(t) * dt  # 积累磨损
            t += dt  # 更新时间
            dt *= 1.1  # 如果误差小，增大时间步长
    return W


# 计算磨损随时间变化的积分计算
def wear_rate(t, T, eta, R):
    integrand = lambda t_prime: (k(t_prime, T, eta, R) * Fn(t_prime) * v(t_prime)) / (H(t_prime) * A(t_prime))
    return adaptive_time_step(integrand, t, initial_step=0.01)


# 计算磨损随时间变化
wear_values = [wear_rate(t, T0, eta0, R0) for t in t_values]

# 绘制各个公式的图
plt.figure(figsize=(10, 6))
plt.plot(t_values, Fn(t_values), label="Normal Force (F_n(t))")
plt.xlabel('Time (s)')
plt.ylabel('Normal Force (N)')
plt.title('Normal Force over Time')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(t_values, v(t_values), label="Sliding Velocity (v(t))")
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Sliding Velocity over Time')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(t_values, H(t_values), label="Hardness (H(t))")
plt.xlabel('Time (s)')
plt.ylabel('Hardness (Pa)')
plt.title('Hardness over Time')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(t_values, A(t_values), label="Contact Area (A(t))")
plt.xlabel('Time (s)')
plt.ylabel('Contact Area (m²)')
plt.title('Contact Area over Time')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(t_values, [k(t, T0, eta0, R0) for t in t_values], label="Wear Coefficient (k(t))")
plt.xlabel('Time (s)')
plt.ylabel('Wear Coefficient (k)')
plt.title('Wear Coefficient over Time')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(t_values, wear_values, label="Cumulative Wear (W(t))")
plt.xlabel('Time (s)')
plt.ylabel('Cumulative Wear (m)')
plt.title('Cumulative Wear over Time')
plt.grid(True)
plt.legend()
plt.show()
