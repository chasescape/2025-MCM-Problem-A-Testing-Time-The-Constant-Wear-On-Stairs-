import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import quad
from functools import partial  # 导入 partial 用于函数参数的绑定

matplotlib.use('TkAgg')  # 或者使用 'Agg' 后端
# 参数设置
m = 60.0  # 物体质量 (kg)
g = 9.81  # 重力加速度 (m/s²)
omega = 2 * np.pi  # 角频率 (rad/年)
v0 = 0.5  # 初始滑动速度 (m/年)
H0 = 1e9  # 初始硬度 (Pa)
A0 = 0.02  # 初始接触面积 (m²)
alpha = 0.1  # 温度对磨损系数的影响系数
beta = 0.05  # 湿度对磨损系数的影响系数
n = 2  # 表面粗糙度的指数
delta = 0.01  # 硬度衰减系数 (1/年)
gamma = 0.01  # 接触面积变化系数 (1/年)
k0 = 1.0  # 基准磨损系数 (无单位)
T0 = 25  # 初始温度 (°C)
eta0 = 50  # 初始湿度 (%)
R0 = 0.2  # 初始粗糙度 (m)
t_values = np.linspace(0, 1, 10)  # 时间数组，范围 0 到 10 年
C = 1.0  # 常数C
D = 0.5  # 常数D
dx = 0.25  # 空间步长 Δx (m)
n0 = 50.0  # 基准湿度 (%)
gamma_T = 0.01  # 温度影响系数
gamma_Y = 0.02  # 湿度影响系数
A = 0.02  # 接触面积 (m²)，现在 A 不再随时间变化
v_func = 0.05  # 速度 (m/年)


# 台阶的长度和宽度
L = 1.5  # 台阶长度
W = 2  # 台阶宽度

# 生成坐标网格
x = np.linspace(0, W, 100)  # X 方向
y = np.linspace(-L, 0, 100)  # Y 方向
X, Y = np.meshgrid(x, y)

# 设置椭圆形高斯分布的标准差
sigma_x = 0.7  # X方向的标准差
sigma_y = 0.5  # Y方向的标准差，减小Y方向的标准差，使边缘的磨损更明显


# 椭圆形高斯分布函数，注意使用衰减因子来增强边缘的磨损
# 高斯分布函数，指定新的中心位置 (x0, y0)
def elliptical_gaussian_distribution_func(X, Y, sigma_x, sigma_y, x0=W/10, y0=L/2):
    return np.exp(-((X - x0) ** 2 / (2 * sigma_x ** 2) + (Y + y0) ** 2 / (2 * sigma_y ** 2)))


# 计算磨损分布
wear_distribution = elliptical_gaussian_distribution_func(X, Y, sigma_x, sigma_y)

# 法向力随时间变化
def Fn(t):
    return m * g * (1 + np.sin(omega * t))  # 单位：N


# 滑动速度随时间变化
def v(t):
    return v0 * np.cos(omega * t)  # 单位：m/年


# 硬度随时间变化
def H(t):
    return H0 * np.exp(-delta * t)  # 单位：Pa


# 假设温度和湿度随时间变化
def T(t):
    return T0 + 10 * np.sin(omega * t)  # 假设温度随时间变化


def n(t):
    return n0 + 10 * np.cos(omega * t)  # 假设湿度随时间变化


# 更新后的磨损系数计算公式
def K(t):
    return k0 * (1 + (T(t) - T0) * gamma_T + (n(t) - n0) * gamma_Y)  # 磨损系数随时间变化


# 计算磨损的积分函数，接受时间 t 和附加的参数 A 和 lambda_
def wear_integrand(t, A, lambda_, m, g, v0, H0, delta, k0, T0, gamma_T, n0, gamma_Y, omega):
    T_t = T(t)
    n_t = n(t)
    Fn_t = Fn(t)  # 法向力
    v_t = v(t)  # 滑动速度
    H_t = H(t)  # 硬度
    K_t = K(t)  # 磨损系数
    # 加入温度影响的扩散效应
    diffusion_effect = 1 + lambda_ * (T_t - T0) / T0
    return Fn_t * v_t * K_t / H_t * A * diffusion_effect


# 计算累积磨损的函数，传递 A 和 lambda_
def calculate_wear(t_max, A, lambda_, m, g, v0, H0, delta, k0, T0, gamma_T, n0, gamma_Y, omega):
    # 使用 partial 将额外的参数传递给 wear_integrand
    integrand = partial(wear_integrand, A=A, lambda_=lambda_, m=m, g=g, v0=v0, H0=H0, delta=delta,
                        k0=k0, T0=T0, gamma_T=gamma_T, n0=n0, gamma_Y=gamma_Y, omega=omega)

    # 使用积分法则计算磨损
    wear = adaptive_time_step(integrand, t_max, initial_step=0.01, tolerance=1e-6)
    return wear


# 自适应时间步长函数，用于控制积分误差
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


# 设置扩散系数和接触面积
lambda_ = 0.08  # 扩散系数
# 假设 t_max 为 10 年，生成时间点 t_values
t_max = 10  # 10年

# 计算并绘制累积磨损随时间变化
wear_values = [calculate_wear(t, A, lambda_, m, g, v0, H0, delta, k0, T0, gamma_T, n0, gamma_Y, omega) for t in
               t_values]

# 计算累积磨损量，并结合高斯分布，生成热力图
W_matrix = np.array(
    [[calculate_wear(t, A, lambda_, m, g, v0, H0, delta, k0, T0, gamma_T, n0, gamma_Y, omega) for t in x] for x in X])

W_total = W_matrix * wear_distribution

# 旋转后的X和Y坐标
X_rot = -Y-1
Y_rot = X

# 使用旋转后的坐标生成椭圆形高斯分布
elliptical_gaussian_distribution_rot = elliptical_gaussian_distribution_func(X_rot, Y_rot, sigma_x, sigma_y)

# 计算旋转后的磨损
W_total_rot = W_matrix * elliptical_gaussian_distribution_rot

# 绘制磨损分布的二维热力图
plt.figure(figsize=(8, 6))
plt.contourf(X_rot, Y_rot, W_total, 50, cmap='viridis')  # X 和 Y 现在代表台阶的长度和宽度
plt.colorbar(label='Wear (W)')
plt.xlabel('Step Length (L)')  # X 轴是台阶的长度
plt.ylabel('Step Width (W)')  # Y 轴是台阶的宽度
plt.title('Wear Distribution on Step Surface')
plt.show()

# 绘制三维图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制三维曲面图
ax.plot_surface(X_rot, Y_rot, W_total, cmap='viridis', edgecolor='none')  # 使用 X 和 Y 坐标

# 添加颜色条
cbar = plt.colorbar(ax.plot_surface(X_rot, Y_rot, W_total, cmap='viridis', edgecolor='none'))
cbar.set_label('Wear (W)')

# 设置标题和轴标签
ax.set_xlabel('Step Length (L)')  # X 轴是台阶的长度
ax.set_ylabel('Step Width (W)')  # Y 轴是台阶的宽度
ax.set_zlabel('Wear (W)')  # 这里是磨损量，作为纵坐标（Z轴）
ax.set_title('3D Wear Distribution on Step Surface')

# 显示图形
plt.show()
