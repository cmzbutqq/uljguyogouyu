import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro
from datetime import datetime
import matplotlib.pyplot as plt
import style

"""
用ARIMA模型预测国家奥运奖牌数
medal_counts_with_host.csv
=>
plots
"""
FREQ = "4YS-JAN"
BEGIN_YEAR = 1992
N_FORECAST = 1

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置中文字体
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号
# SAVE_FOLDER = f"plots/arima/{BEGIN_YEAR}/"
SAVE_FOLDER = f"plots/arima/"
# 读取csv
path = "medal_counts_with_host.csv"
# Rank,NOC,Gold,Silver,Bronze,Total,Year
MEDAL_COUNTS = pd.read_csv(path)
MEDAL_COUNTS = MEDAL_COUNTS[MEDAL_COUNTS["Year"] >= BEGIN_YEAR]


def log_print(str):
    # 写入文件
    with open(SAVE_FOLDER + "log.txt", "a") as f:
        f.write(f"{datetime.now().strftime('%m-%d %H:%M:%S')}:\t{str}\n")
    print(str)


def err_print(str):
    log_print("\n\n\n\n" + str + "\n\n\n")
    with open(SAVE_FOLDER + "err.txt", "a") as f:
        f.write(f"{datetime.now().strftime('%m-%d %H:%M:%S')}:\t{str}\n")


# 选取指定国家
def get_country_data(NOC):
    data_host = MEDAL_COUNTS[MEDAL_COUNTS["NOC"] == NOC]
    # 年份不能重复
    assert (
        not data_host["Year"].duplicated().any()
    ), "a country with MULTIPLE joins in ONE YEAR"
    # 按year排序
    data_host = data_host.sort_values(by="Year", inplace=False)
    # 选择列
    data_host = data_host[["Year", "Total", "Host"]]
    data_host["Year"] = pd.to_datetime(data_host["Year"], format="%Y")
    data_host.set_index("Year", inplace=True)
    # 删除东道主年份
    data = data_host.drop(data_host[data_host["Host"] == 1].index)
    # 给缺少的年份插值
    data_host = data_host.resample(FREQ).interpolate(method="linear")
    data = data.resample(FREQ).interpolate(method="linear")
    # 法国插值至2024
    if NOC == "France":
        data = data_host.copy(deep=True)
        data[-1:] = data[-2:-1]
    # 4年一度
    data_host.index.freq = FREQ
    return data, data_host


# 事前检验：ADF 检验平稳性
def adf_test(series):
    result = adfuller(series)
    print(f"ADF 检验 p 值: {result[1]}")
    return result[1]


# 自动差分直到平稳
def difference_until_stationary(series, max_diff=3):
    diff_series = series.copy()
    diff_count = 0

    while adf_test(diff_series) > 0.05 and diff_count < max_diff:
        diff_series = diff_series.diff().dropna()
        diff_count += 1
        print(f"已进行 {diff_count} 次差分")

    if adf_test(diff_series) <= 0.05:
        print("序列已经平稳")
    else:
        print("差分次数超过最大限制，序列仍然不平稳")

    return diff_series


# 绘制自相关和偏自相关图
def plot_acf_pacf(series,country):
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plot_acf(series, ax=plt.gca())
    plt.title(f"ACF of {country}")

    plt.subplot(122)
    plot_pacf(series, ax=plt.gca())
    plt.title(f"PACF of {country}")

    return fig


# 事后检验：残差分析
def residual_analysis(model, series,country):
    residuals = model.resid
    resid_fig = plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plot_acf(residuals, ax=plt.gca())
    plt.title(f"ACF of {country} residual")

    plt.subplot(122)
    plot_pacf(residuals, ax=plt.gca())
    plt.title(f"PACF of {country} residual")

    # QQ 图
    qq_fig = qqplot(residuals, line="s")
    plt.title(f"qqplot of {country} residual")

    # 正态性检验
    _, p_value = shapiro(residuals)
    print(f"Shapiro-Wilk 正态性检验 p 值: {p_value}")
    if p_value > 0.05:
        print("残差序列接近正态分布")
    else:
        print("残差序列不服从正态分布")

    return resid_fig, qq_fig


# 自动选择 ARIMA 模型的 p 和 q 值
def auto_select_pq(series, max_p=5, max_q=5):
    best_aic = np.inf
    best_order = None
    best_model = None

    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            try:
                model = ARIMA(series, order=(p, 1, q))  # 已经差分过，d=1
                model_fit = model.fit()
                aic = model_fit.aic
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, 1, q)
                    best_model = model_fit
            except Exception as e:
                continue  # 若模型拟合失败则跳过

    print(f"最优模型的阶数为 p={best_order[0]}, d=1, q={best_order[2]}，AIC={best_aic}")
    return best_model, best_order


# 预测并绘制图表
def forecast_and_plot(
    series, model, n_forecast, diff_series, diff_count, real_series, country
):
    # 获取预测结果及其置信区间
    forecast_result = model.get_forecast(steps=n_forecast)
    forecast = forecast_result.predicted_mean  # 预测值
    conf_int_99 = forecast_result.conf_int(alpha=0.01)  # 99% 置信区间
    conf_int_95 = forecast_result.conf_int(alpha=0.05)  # 95% 置信区间
    conf_int_90 = forecast_result.conf_int(alpha=0.10)  # 90% 置信区间

    if diff_count > 0:
        for i in range(diff_count):
            forecast = forecast.cumsum()  # 累加每次差分的结果
            forecast += diff_series.iloc[-1]  # 加上最后一个差分值
        # 差分值改成原值
        forecast -= diff_series.iloc[-1]
        forecast += series.iloc[-1]

        for i in range(diff_count):
            conf_int_99 = conf_int_99.cumsum()  # 累加每次差分的结果
            conf_int_99 += diff_series.iloc[-1]  # 加上最后一个差分值
        # 差分值改成原值
        conf_int_99 -= diff_series.iloc[-1]
        conf_int_99 += series.iloc[-1]

        for i in range(diff_count):
            conf_int_95 = conf_int_95.cumsum()  # 累加每次差分的结果
            conf_int_95 += diff_series.iloc[-1]  # 加上最后一个差分值
        # 差分值改成原值
        conf_int_95 -= diff_series.iloc[-1]
        conf_int_95 += series.iloc[-1]

        for i in range(diff_count):
            conf_int_90 = conf_int_90.cumsum()  # 累加每次差分的结果
            conf_int_90 += diff_series.iloc[-1]  # 加上最后一个差分值
        # 差分值改成原值
        conf_int_90 -= diff_series.iloc[-1]
        conf_int_90 += series.iloc[-1]

    H = np.float64(1.304798962386511)  # 来自calc_host_buff.py
    if country == "United States":
        forecast[0:1] *= H
        conf_int_99[0:1] *= H
        conf_int_95[0:1] *= H
        conf_int_90[0:1] *= H

    # 绘图部分
    # 绘制实际值的折线图
    fig = plt.figure(figsize=(10, 6))
    plt.plot(real_series, label="实际值", color="blue")

    plt.plot(series, label="非东道主时的值", color="blue", linestyle="--")
    # type(forecast_cumsum) <class 'pandas.core.series.Series'>
    # forecast_cumsum -> idx:2028-01-01    val:123.079235 ...

    # 在开头添加数据点 连上实际的折线
    forecast = pd.concat([series[-1:], forecast])
    plt.plot(forecast, label="预测值", color="red")

    # 绘制置信区间
    plt.fill_between(
        forecast.index,
        pd.concat([series[-1:], conf_int_99.iloc[:, 0]]),
        pd.concat([series[-1:], conf_int_99.iloc[:, 1]]),
        color="lightblue",
        alpha=0.1,
        label="99% 置信区间",
    )
    plt.fill_between(
        forecast.index,
        pd.concat([series[-1:], conf_int_95.iloc[:, 0]]),
        pd.concat([series[-1:], conf_int_95.iloc[:, 1]]),
        color="blue",
        alpha=0.1,
        label="95% 置信区间",
    )
    plt.fill_between(
        forecast.index,
        pd.concat([series[-1:], conf_int_90.iloc[:, 0]]),
        pd.concat([series[-1:], conf_int_90.iloc[:, 1]]),
        color="grey",
        alpha=0.2,
        label="90% 置信区间",
    )

    plt.title(f"ARIMA Prediction of {country}")
    plt.legend()
    return fig


# 生成模拟时间序列数据
def generate_test_series():
    np.random.seed(0)
    # 生成一个时间序列数据，包含趋势和随机噪声
    dates = pd.date_range(start="2020-01-01", periods=100, freq=FREQ)
    trend = np.linspace(10, 50, 100)  # 线性趋势
    noise = np.random.normal(0, 5, size=100)  # 随机噪声
    series = trend + noise
    data = pd.DataFrame({"Value": series}, index=dates)
    return data


# 主程序
def main(country):
    data, real_data = get_country_data(country)
    # 选择要建模的序列（例如：'Value' 列）
    series = data["Total"]
    if len(series) < 2:
        log_print(f"\n\n{country} TOO SMALL SAMPLES\n")
        return

    # 事前检验：平稳性检验，自动差分
    print("初始序列平稳性检验：")
    if adf_test(series) > 0.05:
        print("序列非平稳，正在进行差分...")
        diff_series = difference_until_stationary(series)
        diff_count = 1  # 标记差分次数
    else:
        diff_series = series
        diff_count = 0

    # 绘制自相关和偏自相关图，帮助选择 ARIMA 模型参数
    fig1 = plot_acf_pacf(diff_series,country)

    # 自动选择 p, q 值
    best_model, best_order = auto_select_pq(diff_series, max_p=5, max_q=5)

    # 输出最优模型
    print(f"最优模型：ARIMA{best_order}")

    # 预测并绘制结果
    real_series = real_data["Total"]
    fig2 = forecast_and_plot(
        series,
        best_model,
        n_forecast=N_FORECAST,
        diff_series=diff_series,
        diff_count=diff_count,
        real_series=real_series,
        country=country,
    )

    # 事后检验：残差分析
    fig3, fig4 = residual_analysis(best_model, series,country)

    fig1.savefig(SAVE_FOLDER + f"{country}_{BEGIN_YEAR}_{N_FORECAST}_acf-pacf.png")
    fig2.savefig(SAVE_FOLDER + f"{country}_{BEGIN_YEAR}_{N_FORECAST}_forecast.png")
    fig3.savefig(
        SAVE_FOLDER + f"{country}_{BEGIN_YEAR}_{N_FORECAST}_resid-acf-pacf.png"
    )
    fig4.savefig(SAVE_FOLDER + f"{country}_{BEGIN_YEAR}_{N_FORECAST}_qqplot.png")


if __name__ == "__main__":
    # countries=MEDAL_COUNTS['NOC'].unique()
    # 高绩效-均衡型，美国、中国、德国、法国，中等绩效-稳定型，意大利、加拿大、西班牙、瑞典奖牌中等且波动小。这两类
    countries = (
        "France",
        "China",
        "United States",
        "Germany",
        "France",
        "Italy",
        "Canada",
        "Spain",
        "Sweden",
    )
    log_print(f"{len(countries)=}")
    for i, NOC in enumerate(countries):
        log_print(f"\t{i}\tTRYING {NOC=}:")
        # main(NOC)
        try:
            main(NOC)
        except Exception as e:
            err_print(f"ERROR in {NOC=}: {e}, {type(e)=}")
        plt.close("all")
