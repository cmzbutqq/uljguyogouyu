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


FREQ = "4YS-JAN"
BEGIN_YEAR = 1992
# Rank,NOC,Gold,Silver,Bronze,Total,Year
MEDAL_COUNTS = pd.read_csv("medal_counts_with_host.csv")
MEDAL_COUNTS = MEDAL_COUNTS[MEDAL_COUNTS["Year"] >= BEGIN_YEAR]


# 获取国家历年成绩
def get_country_medals(NOC):
    data_real = MEDAL_COUNTS[MEDAL_COUNTS["NOC"] == NOC]
    # 年份不能重复
    assert (
        not data_real["Year"].duplicated().any()
    ), "a country with MULTIPLE joins in ONE YEAR"
    # 按year排序
    data_real = data_real.sort_values(by="Year", inplace=False)
    # 列操作
    data_real = data_real[["Year", "Total", "Host"]]
    data_real["Year"] = pd.to_datetime(data_real["Year"], format="%Y")
    data_real.set_index("Year", inplace=True)
    # 删除东道主年份
    data_interp = data_real.drop(data_real[data_real["Host"] == 1].index)
    # 给缺少的年份插值
    data_real = data_real.resample(FREQ).interpolate(method="linear")
    data_interp = data_interp.resample(FREQ).interpolate(method="linear")
    # 法国插值至2024
    if NOC == "France":
        data_interp = data_real.copy(deep=True)
        data_interp[-1:] = data_interp[-2:-1]
    # index转为列
    data_real.reset_index(inplace=True)
    data_interp.reset_index(inplace=True)
    # timestamp转数字
    data_real["Year"] = data_real["Year"].map(lambda x: x.year)
    data_interp["Year"] = data_interp["Year"].map(lambda x: x.year)
    return data_interp, data_real


# ======================
# 1. 数据准备与预处理
# ======================


def main(country):
    data, _ = get_country_medals(country)
    data = data[["Year", "Total"]]
    data = data.rename(columns={"Total": "medals", "Year": "year"})
    data.set_index("year", inplace=True)

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
    arima_model, arch_result = train_arima_arch(
        data, arima_order, arch_lags
    )  # 变量名修改

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
    H = np.sqrt(np.float64(1.304798962386511))  # 来自calc_host_buff.py
    if country == "United States":
        mean_pred *= H
        lower_bound *= H
        upper_bound *= H

    print(f"\n预测结果 {forecast_year}:")
    print(f"均值预测: {mean_pred:.1f} 枚")
    print(f"95%置信区间: [{lower_bound:.1f}, {upper_bound:.1f}]")

    # ======================
    # 4. 可视化结果
    # ======================
    plt.figure(figsize=(12, 6))
    plt.title(f"ARIMA-ARCH co-predict: total medal count for {country}")
    plt.plot(data.index, data["medals"], "bo-", label="actual")
    plt.plot(forecast_year, mean_pred, "ro", label="predicted")
    plt.fill_between(
        [forecast_year - 1, forecast_year + 1],
        lower_bound,
        upper_bound,
        color="red",
        alpha=0.1,
        label="95% conf_int",
    )
    plt.legend()
    plt.grid(True)
    plt.xlabel("Year")
    plt.ylabel("Medal Count")
    plt.savefig(f"plots/arima-arch/co-predict_{country}.png", dpi=300)
    # plt.show()

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


if __name__ == "__main__":
    countries = MEDAL_COUNTS["NOC"].unique()

    # countries = (
    #     "China",
    #     "Australia",
    #     "Japan",
    #     "Great Britain",
    #     "United States",
    #     "France",
    #     "Canada",
    #     "Germany",
    #     "Kenya",
    #     "Jamaica",
    # )
    for c in countries:
        try:
            main(c)
        except Exception as e:
            print(f"Error for {c}: {e}")
            continue
        
