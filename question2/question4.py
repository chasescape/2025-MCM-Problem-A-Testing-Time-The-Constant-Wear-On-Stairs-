import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib

matplotlib.use('TkAgg')  # 或者使用 'Agg' 后端

# 随机种子，保证每次生成相同的数据
np.random.seed(42)

# 生成模拟数据
n_samples = 100  # 生成100个样本

# 假设数据中有以下变量：
# 1. 楼梯年龄 (years) [从1到50年]
# 2. 使用频率 (frequency) [0到10之间的随机数，表示使用频繁程度]
# 3. 建筑结构类型 (structure_type) [1=普通结构，2=加固结构，3=豪华结构]
# 4. 日常活动模式 (activity_level) [0到1之间，0表示无活动，1表示高频活动]
# 5. 楼梯磨损 (wear) [真实磨损值，单位为米]

# 1. 楼梯年龄 (years)
stair_age = np.random.randint(1, 51, size=n_samples)

# 2. 使用频率 (frequency)，这里假设高频使用会增加磨损
usage_frequency = np.random.uniform(0, 10, size=n_samples)

# 3. 建筑结构类型：假设不同结构的承载能力和耐久度不同
structure_type = np.random.choice([1, 2, 3], size=n_samples)  # 1: 普通，2: 加固，3: 豪华

# 4. 日常活动模式：假设高活动强度会加剧磨损
activity_level = np.random.uniform(0, 1, size=n_samples)

# 5. 磨损（生成一个粗略的公式，考虑到年龄、使用频率、结构类型和活动水平）
wear = 0.1 * stair_age + 0.5 * usage_frequency + 0.3 * (structure_type == 1) + 0.2 * (
        structure_type == 2) + 0.1 * activity_level
wear += np.random.normal(0, 1, size=n_samples)  # 添加一些噪声

# 将数据存储到一个DataFrame中
data = pd.DataFrame({
    'Stair_Age': stair_age,
    'Usage_Frequency': usage_frequency,
    'Structure_Type': structure_type,
    'Activity_Level': activity_level,
    'Wear': wear
})

# 查看数据的一部分
print(data.head())

# 使用 statsmodels.formula.api 来构建模型
# 使用结构类型作为分类变量，直接使用一个公式模型
model_sm = smf.ols('Wear ~ Stair_Age + Usage_Frequency + C(Structure_Type) + Activity_Level', data=data).fit()

# 预测磨损值
y_pred = model_sm.predict(data)

# 计算均方误差 (MSE) 来评估模型的表现
mse = np.mean((data['Wear'] - y_pred) ** 2)
print(f'Mean Squared Error: {mse}')

# 绘制预测磨损和实际磨损的比较
plt.figure(figsize=(10, 6))
plt.scatter(data['Wear'], y_pred, color='blue', label='Predicted vs Actual')
plt.plot([min(data['Wear']), max(data['Wear'])], [min(data['Wear']), max(data['Wear'])], color='red', linestyle='--',
         label='Perfect Prediction')
plt.xlabel('Actual Wear')
plt.ylabel('Predicted Wear')
plt.title('Predicted vs Actual Wear')
plt.legend()
plt.grid(True)
plt.show()

# 查看回归系数
print("Regression Coefficients:")
print(model_sm.params)

# 方差分析 (ANOVA)
anova_table = sm.stats.anova_lm(model_sm, typ=2)
print("ANOVA Table:")
print(anova_table)

# 置信区间
conf_int = model_sm.conf_int(alpha=0.05)  # 95% 置信区间
print("Confidence Intervals for Coefficients:")
print(conf_int)

# 提取系数和置信区间
coefficients = model_sm.params
lower_bounds = conf_int[0]
upper_bounds = conf_int[1]

# 绘制置信区间图
plt.figure(figsize=(10, 6))
plt.bar(coefficients.index, coefficients, yerr=[coefficients - lower_bounds, upper_bounds - coefficients], capsize=5,
        color='lightblue')
plt.xlabel('Coefficient')
plt.ylabel('Value')
plt.title('Regression Coefficients with 95% Confidence Intervals')
plt.grid(True)
plt.show()

# --- 绘制方差贡献图 ---
# 提取各个变量对方差的贡献
anova_contrib = anova_table['sum_sq'] / anova_table['sum_sq'].sum() * 100  # 转换为百分比

# 绘制方差贡献图
plt.figure(figsize=(10, 6))
anova_contrib.plot(kind='bar', color='orange')
plt.xlabel('Variable')
plt.ylabel('Variance Contribution (%)')
plt.title('Variance Contribution of Each Variable')
plt.grid(True)
plt.show()

# --- 绘制磨损随楼梯年龄、使用频率、活动水平的关系 ---
plt.figure(figsize=(10, 6))
plt.scatter(data['Stair_Age'], data['Wear'], label='Wear vs Age', color='green')
plt.xlabel('Stair Age (years)')
plt.ylabel('Wear (m)')
plt.title('Wear vs Stair Age')
plt.grid(True)
plt.legend()
plt.show()
