import numpy as np
import pandas as pd
from functools import partial

# 参数设定
g = 9.81  # 重力加速度，单位 m/s²
v0 = 1.5  # 初始滑动速度，单位 m/年
H0 = 150  # 初始硬度，单位 Pa
delta = 0.02  # 硬度衰减率
k0 = 0.01  # 初始磨损系数
T0 = 20  # 初始温度，单位 °C
gamma_T = 0.005  # 温度对磨损的影响系数
n0 = 60  # 初始湿度，单位 %
gamma_Y = 0.002  # 湿度对磨损的影响系数
omega = 2 * np.pi / 365  # 基于日变化的角频率

# 数据集
data = {
    'Weight': [68.23, 52.71, 72.56, 61.44, 84.11, 45.98, 63.25, 78.49, 57.38, 50.12],
    'Humidity': [56.14, 75.03, 62.78, 65.82, 58.39, 50.91, 68.12, 59.47, 66.88, 70.23],
    'Temperature': [22.68, 18.46, 30.02, 27.11, 21.58, 16.42, 19.03, 28.34, 23.92, 17.85],
    'Initial_Hardness': [180.52, 130.61, 145.23, 175.9, 162.85, 115.67, 150.39, 160.82, 140.95, 135.14],
    'Stride_Frequency': [120.34, 98.12, 140.72, 115.56, 130.42, 105.21, 130.15, 110.29, 125.51, 105.77],
    'Sliding_Speed': [1.25, 1.84, 1.38, 0.92, 1.49, 1.15, 1.77, 1.32, 1.03, 1.62],
    'Stair_Age': [25, 5, 15, 10, 20, 3, 12, 18, 22, 8],
    'Usage_Mode': [1, 0, 2, 1, 2, 0, 1, 2, 1, 0],
    'Usage_Frequency': [5.23, 2.56, 8.97, 6.78, 9.43, 3.12, 4.65, 7.43, 5.89, 3.98],
    'User_Count': [23, 10, 45, 32, 18, 7, 25, 38, 29, 15],
    'Measured_Wear': [10.78, 4.37, 14.65, 12.29, 13.52, 6.21, 9.51, 11.43, 10.03, 5.72]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# 假设接触面积与体重成比例
k = 0.1  # 接触面积与体重的比例系数


# 定义磨损计算函数
def Fn(t, m):
    return m * g * (1 + np.sin(omega * t))  # 单位：N


# 滑动速度随时间变化
def v(t, v0):
    return v0 * np.cos(omega * t)  # 单位：m/年


# 硬度随时间变化
def H(t, H0, delta):
    return H0 * np.exp(-delta * t)  # 单位：Pa


# 假设温度和湿度随时间变化
def T(t, T0, omega):
    return T0 + 10 * np.sin(omega * t)  # 假设温度随时间变化


def n(t, n0, omega):
    return n0 + 10 * np.cos(omega * t)  # 假设湿度随时间变化


# 更新后的磨损系数计算公式
def K(t, T_t, n_t, T0, gamma_T, n0, gamma_Y, k0):
    return k0 * (1 + (T_t - T0) * gamma_T + (n_t - n0) * gamma_Y)  # 磨损系数随时间变化


# 计算磨损的积分函数，接受时间 t 和附加的参数 A 和 lambda_
def wear_integrand(t, A, lambda_, m, g, v0, H0, delta, k0, T0, gamma_T, n0, gamma_Y, omega):
    T_t = T(t, T0, omega)
    n_t = n(t, n0, omega)
    Fn_t = Fn(t, m)  # 法向力
    v_t = v(t, v0)  # 滑动速度
    H_t = H(t, H0, delta)  # 硬度
    K_t = K(t, T_t, n_t, T0, gamma_T, n0, gamma_Y, k0)  # 磨损系数
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


# 设置扩散系数
lambda_ = 0.08  # 扩散系数
t_max = 10  # 10年

# 遍历 DataFrame 的每一行，计算每一行的磨损量
for index, row in df.iterrows():
    Weight = row['Weight']
    Humidity = row['Humidity']
    Temperature = row['Temperature']
    Initial_Hardness = row['Initial_Hardness']
    Stride_Frequency = row['Stride_Frequency']
    Sliding_Speed = row['Sliding_Speed']
    Stair_Age = row['Stair_Age']
    Usage_Mode = row['Usage_Mode']
    Usage_Frequency = row['Usage_Frequency']
    User_Count = row['User_Count']
    Measured_Wear = row['Measured_Wear']

    # 计算接触面积 A
    A = k * Weight  # 假设接触面积与体重成比例

    # 计算磨损量
    wear = calculate_wear(t_max=10, A=A, lambda_=lambda_, m=Weight, g=g, v0=Sliding_Speed, H0=Initial_Hardness,
                          delta=delta,
                          k0=k0, T0=T0, gamma_T=gamma_T, n0=n0, gamma_Y=gamma_Y, omega=omega)

    print(f"第 {index} 行 - 估算磨损量: {wear:.2f} 单位")


