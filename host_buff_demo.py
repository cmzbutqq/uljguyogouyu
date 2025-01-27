import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import style

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
    if NOC == "Spain":
        data_interp = data_real.copy(deep=True)
        data_interp[0:1] = data_interp[1:2]
    # # index转为列
    # data_real.reset_index(inplace=True)
    # data_interp.reset_index(inplace=True)
    # # timestamp转数字
    # data_real["Year"] = data_real["Year"].map(lambda x: x.year)
    # data_interp["Year"] = data_interp["Year"].map(lambda x: x.year)
    return data_interp, data_real


# 预测并绘制图表
def ploot(series, real_series, country):

    # 绘图部分
    # 绘制实际值的折线图
    fig = plt.figure(figsize=(10, 6))
    plt.plot(series, label="host interped", linestyle="--")
    plt.plot(real_series, label="actual")

    plt.title(f"Performance of {country}")
    plt.legend()
    plt.savefig(f"plots/host_effect_demo/{country}_{BEGIN_YEAR}.png")
    return fig


if __name__ == "__main__":

    countries = MEDAL_COUNTS[MEDAL_COUNTS['Host']==True]["NOC"].unique()
    # 高绩效-均衡型，美国、中国、德国、法国，中等绩效-稳定型，意大利、加拿大、西班牙、瑞典奖牌中等且波动小。这两类

    for c in countries:
        data_interp, data_real = get_country_medals(c)
        ploot(data_interp["Total"], data_real["Total"], c)
        plt.close("all")
