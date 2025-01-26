import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FREQ = "4YS-JAN"
BEGIN_YEAR = 1992
N_FORECAST = 1
import matplotlib.pyplot as plt

# SAVE_FOLDER = f"plots/arima/{BEGIN_YEAR}/"
SAVE_FOLDER = f"plots/arima/"
# 读取csv
path = "medal_counts_with_host.csv"
# Rank,NOC,Gold,Silver,Bronze,Total,Year
MEDAL_COUNTS = pd.read_csv(path)
MEDAL_COUNTS = MEDAL_COUNTS[MEDAL_COUNTS["Year"] >= BEGIN_YEAR]


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


# 选取指定国家
def get_country_host_buff(NOC):
    data, host = get_country_data(NOC)
    data = data[host["Host"] == True]
    host = host[host["Host"] == True]
    data = data["Total"].sum()
    host = host["Total"].sum()
    return host, data


if __name__ == "__main__":
    countries = MEDAL_COUNTS["NOC"].unique()
    A, B = [], []
    for c in countries:
        buffed, unbuffed = get_country_host_buff(c)
        if unbuffed == 0 and buffed == 0:
            continue
        A.append(buffed)
        B.append(unbuffed)

    # plt.plot(A,'o')
    # plt.plot(B,'o')
    # plt.plot(np.array(A)/np.array(B),'o')
    # plt.show()

    H = np.array(A).sum() / np.array(B).sum()
    print(f"{H=}")  # H=np.float64(1.304798962386511)
