# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from scipy import stats
from sklearn.metrics import mean_absolute_error

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 正常显示中文
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号


# ======================
# 1. 数据准备与预处理
# ======================
def generate_sample_data():
    """生成示例数据（巴西1992-2024奖牌数）"""
    years = np.arange(1992, 2025, 4)  # 奥运会年份
    medals = [5, 8, 10, 15, 28, 12, 10, 18, 7]  # 虚构数据

    df = pd.DataFrame({"year": years, "medals": medals}).set_index("year")
    return df


data = generate_sample_data()
print("原始数据:\n", data)


# ======================
# 2. 构建ARIMA-ARCH模型
# ======================
def train_arima_arch(data, arima_order=(1, 1, 1), arch_lags=1):
    """训练ARIMA-ARCH联合模型"""
    # 训练ARIMA模型
    arima = ARIMA(endog=data["medals"], order=arima_order).fit()

    # 提取标准化残差并去NaN
    residuals = arima.resid / arima.resid.std()
    residuals = residuals.dropna()

    # 训练ARCH模型
    arch = arch_model(residuals, vol="ARCH", p=arch_lags, dist="StudentsT").fit(
        disp="off"
    )

    return arima, arch


# 训练模型
arima_order = (1, 1, 1)
arch_lags = 1
arima_model, arch_result = train_arima_arch(data, arima_order, arch_lags)  # 变量名修改

print("\nARIMA模型摘要:")
print(arima_model.summary())
print("\nARCH模型摘要:")
print(arch_result.summary())  # 变量名修改


# ======================
# 3. 预测与置信区间计算
# ======================
def predict_with_uncertainty(model_arima, model_arch, horizon=1, ci=0.95):
    """联合预测均值与波动性"""
    # 预测均值
    mean_forecast = model_arima.forecast(steps=horizon)

    # 预测波动性
    arch_forecast = model_arch.forecast(horizon=horizon)
    variance = arch_forecast.variance.iloc[-1, :].values

    # 计算置信区间
    z_score = np.abs(stats.norm.ppf((1 - ci) / 2))
    lower = mean_forecast.iloc[0] - z_score * np.sqrt(variance[0])
    upper = mean_forecast.iloc[0] + z_score * np.sqrt(variance[0])

    return mean_forecast.iloc[0], lower, upper


# 预测2028年
forecast_year = 2028
mean_pred, lower_bound, upper_bound = predict_with_uncertainty(
    arima_model, arch_result  # 使用修正后的变量名
)

print(f"\n预测结果 {forecast_year}:")
print(f"均值预测: {mean_pred:.1f} 枚")
print(f"95%置信区间: [{lower_bound:.1f}, {upper_bound:.1f}]")

# ======================
# 4. 可视化结果
# ======================
plt.figure(figsize=(12, 6))
plt.title("ARIMA-ARCH联合预测: 巴西奥运奖牌数")
plt.plot(data.index, data["medals"], "bo-", label="实际值")
plt.plot(forecast_year, mean_pred, "ro", label="预测均值")
plt.fill_between(
    [forecast_year - 1, forecast_year + 1],
    lower_bound,
    upper_bound,
    color="red",
    alpha=0.1,
    label="95%置信区间",
)
plt.legend()
plt.grid(True)
plt.xlabel("年份")
plt.ylabel("奖牌数")
plt.show()


# ======================
# 5. 模型评估（滚动预测）
# ======================
def rolling_validation(data, train_size=5):
    """滚动窗口验证模型性能"""
    mae_scores = []
    actuals, preds = [], []

    for i in range(train_size, len(data)):
        train = data.iloc[:i]
        test = data.iloc[i]

        arima, arch = train_arima_arch(train)
        pred, _, _ = predict_with_uncertainty(arima, arch)

        actuals.append(test["medals"])
        preds.append(pred)  # 直接添加标量
        mae = mean_absolute_error([test["medals"]], [pred])  # 参数转为列表
        mae_scores.append(mae)

    return actuals, preds, mae_scores


# 执行滚动验证
actuals, preds, mae_scores = rolling_validation(data)

print("\n滚动验证MAE:", np.round(mae_scores, 2))
print("平均MAE:", np.mean(mae_scores).round(2))
