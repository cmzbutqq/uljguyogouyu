import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro
from datetime import datetime

FREQ = "4YS-JAN"
BEGIN_YEAR = 1992
N_FORECAST = 1
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置中文字体
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号
# SAVE_FOLDER = f"plots/arima/{BEGIN_YEAR}/"
SAVE_FOLDER = f"plots/arima/"
# 读取csv
path = "medal_counts_with_host.csv"
# Rank,NOC,Gold,Silver,Bronze,Total,Year
MEDAL_COUNTS = pd.read_csv(path)
MEDAL_COUNTS = MEDAL_COUNTS[MEDAL_COUNTS["Year"] >= BEGIN_YEAR]

ATHLETES = pd.read_csv("2025_Problem_C_Data/summerOly_athletes.csv")
ATHLETES = ATHLETES[ATHLETES["Year"] >= BEGIN_YEAR]


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


def get_athlete_data(Team, year):
    records = ATHLETES[(ATHLETES["Year"] == year) & (ATHLETES["Team"] == Team)]

    # 每个项目参加的人次数
    sports = records.groupby("Sport")["Medal"].count()
    # 最多的前三个大项目，参与这三个大项目的人次是adventage_athletes,然后参加别的项目的是other_athletes
    advantage_athletes = sports.nlargest(3).sum()
    other_athletes = sports.sum() - advantage_athletes

    medals, _ = get_country_medals(Team)
    total_medals = medals[medals["Year"] == year]["Total"].sum()
    return advantage_athletes, other_athletes, total_medals


def make_csv():
    with open(f"country-year_analysis.csv", "w") as f:
        f.writelines("Country,Year,Advantage_athletes,Other_athletes,Total_medals\n")
        for country in MEDAL_COUNTS["NOC"].unique():
            for year in MEDAL_COUNTS[MEDAL_COUNTS["NOC"] == country]["Year"].unique():
                top_sports, other_sports, total_medals = get_athlete_data(country, year)
                f.writelines(
                    f"{country},{year},{top_sports},{other_sports},{total_medals}\n"
                )


if __name__ == "__main__":
    make_csv()
