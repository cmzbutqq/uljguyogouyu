import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import shapiro

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置中文字体
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

# ====================
# 1. 数据准备
# ====================

country = "South Korea"
data = pd.read_csv(f"country-year_analysis.csv")
df = data[data["Country"] == country]

# ====================
# 2. 数据预处理
# ====================
X = df[["Advantage_athletes", "Other_athletes", "Focus"]]
y = df["Total_medals"]

# ====================
# 3. 模型训练与预测
# ====================
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# ====================
# 4. 模型评估
# ====================
print("=== 模型评估 ===")
print(f"R² Score: {r2_score(y, y_pred):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.2f}")

# ====================
# 5. 回归方程输出
# ====================
print("\n=== 回归方程 ===")
equation = f"总奖牌数 = {model.intercept_:.2f} "
for i, coef in enumerate(model.coef_):
    equation += f"+ {coef:.3f}×{X.columns[i]} "
print(equation)

# ====================
# 6. 可视化分析
# ====================
# # 3D散点图与回归平面
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection="3d")

# # 绘制实际数据点
# ax.scatter(X.iloc[:, 0], X.iloc[:, 1], y, c="r", marker="o", s=50, label="实际值")

# # 生成预测平面
# x1_range = np.linspace(X.iloc[:, 0].min(), X.iloc[:, 0].max(), 10)
# x2_range = np.linspace(X.iloc[:, 1].min(), X.iloc[:, 1].max(), 10)
# x1, x2 = np.meshgrid(x1_range, x2_range)
# y_plane = model.predict(np.c_[x1.ravel(), x2.ravel()]).reshape(x1.shape)

# # 绘制回归平面
# surf = ax.plot_surface(x1, x2, y_plane, alpha=0.6, cmap="viridis")
# fig.colorbar(surf, shrink=0.5, aspect=5)

# ax.set_xlabel("优势项目运动员")
# ax.set_ylabel("其他项目运动员")
# ax.set_zlabel("总奖牌数")
# plt.title("二元线性回归平面")
# plt.legend()
# plt.show()

# # 残差分析图
# plt.figure(figsize=(8, 4))
residuals = y - y_pred
# plt.scatter(y_pred, residuals, c="blue")
# plt.axhline(y=0, color="red", linestyle="--")
# plt.xlabel("预测值")
# plt.ylabel("残差")
# plt.title("残差分布图")
# plt.show()

# ====================
# 7. 统计诊断
# ====================
# 正态性检验
_, p_value = shapiro(residuals)
print(f"\n=== 统计诊断 ===")
print(f"残差正态性检验p值: {p_value:.3f}")

# 共线性检查
# correlation = X.corr().iloc[0,1]
# print(f"自变量相关系数: {correlation:.3f}")

# ====================
# 8. 新数据预测
# ====================
new_data = [[48, 92, 0.5]]  # 2024年示例数据
prediction = model.predict(new_data)
print(f"\n=== 2024年预测 ===")
print(f"优势运动员: {new_data[0][0]}人")
print(f"其他运动员: {new_data[0][1]}人")
print(f"预测奖牌数: {prediction[0]:.1f}枚")
