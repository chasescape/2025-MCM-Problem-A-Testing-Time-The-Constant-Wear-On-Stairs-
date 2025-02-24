import numpy as np
import matplotlib.pyplot as plt
import matplotlib


matplotlib.use('TkAgg')  # 或者使用 'Agg' 后端
# 台阶的长度和宽度
L = 1.5  # 台阶长度
W = 0.8  # 台阶宽度

# 生成坐标网格
x = np.linspace(0, W, 200)  # X 方向
y = np.linspace(-L, L, 200)  # Y 方向
X, Y = np.meshgrid(x, y)

# 设置椭圆形高斯分布的标准差
sigma_x = 0.5  # X方向的标准差
sigma_y = 0.5  # Y方向的标准差


# 椭圆形高斯分布函数
def elliptical_gaussian_distribution_func(X, Y, sigma_x, sigma_y):
    return np.exp(-((X ** 2 / (2 * sigma_x ** 2)) + (Y ** 2 / (2 * sigma_y ** 2))))


# 使用高斯分布生成步伐的概率密度
distribution = elliptical_gaussian_distribution_func(X, Y, sigma_x, sigma_y)

# 归一化概率分布（使得总和为1）
distribution /= np.sum(distribution)

# 定义生成脚步的数量
num_steps_one_person = 1000  # 一个人的步伐数
num_steps_two_people = 2000  # 两个人的步伐数


# 从高斯分布中随机采样步伐位置
def generate_steps(distribution, num_steps):
    # 展平分布为一维
    flat_distribution = distribution.flatten()
    # 使用多项式分布来根据概率采样步伐位置
    chosen_indices = np.random.choice(np.arange(flat_distribution.size), size=num_steps, p=flat_distribution)

    # 获取对应的X, Y坐标
    steps_x = X.flatten()[chosen_indices]
    steps_y = Y.flatten()[chosen_indices]

    return steps_x, steps_y


# 一个人人走的步伐
x_one_person, y_one_person = generate_steps(distribution, num_steps_one_person)

# 两个人走的步伐
x_two_people_1, y_two_people_1 = generate_steps(distribution, num_steps_two_people)
x_two_people_2, y_two_people_2 = generate_steps(distribution, num_steps_two_people)

# 绘制一个人走的散点图
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 1)
plt.scatter(x_one_person, y_one_person, color='blue', label='Footsteps for One Person', s=10)
plt.xlim(0, W)
plt.ylim(-L, L)
plt.xlabel('Step Length (L)')
plt.ylabel('Step Width (W)')
plt.title('Footsteps for One Person')
plt.legend()

# 绘制两个人走的散点图
plt.subplot(1, 2, 2)
plt.scatter(x_two_people_1, y_two_people_1, color='blue', label='Footsteps for Person 1', s=10)
plt.scatter(x_two_people_2, y_two_people_2, color='red', label='Footsteps for Person 2', s=10)
plt.xlim(0, W)
plt.ylim(-L, L)
plt.xlabel('Step Length (L)')
plt.ylabel('Step Width (W)')
plt.title('Footsteps for Two People')
plt.legend()

# 显示图形
plt.tight_layout()
plt.show()
