import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro
from datetime import datetime
FREQ='4YS-JAN'
BEGIN_YEAR = 1992
N_FORECAST=1
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号
# SAVE_FOLDER = f"plots/arima/{BEGIN_YEAR}/"
SAVE_FOLDER = f"plots/arima/"
#读取csv
path="medal_counts_with_host.csv"
# Rank,NOC,Gold,Silver,Bronze,Total,Year
MEDAL_COUNTS = pd.read_csv(path)
MEDAL_COUNTS=MEDAL_COUNTS[MEDAL_COUNTS['Year']>=BEGIN_YEAR] 

#选取指定国家
def get_country_data(NOC):
    data_host = MEDAL_COUNTS[MEDAL_COUNTS['NOC']==NOC]
    #年份不能重复
    assert not data_host['Year'].duplicated().any(),"a country with MULTIPLE joins in ONE YEAR"
    # 按year排序
    data_host = data_host.sort_values(by='Year',inplace=False)
    # 选择列
    data_host = data_host[['Year','Total','Host']]
    data_host['Year'] = pd.to_datetime(data_host['Year'], format='%Y')
    data_host.set_index('Year', inplace=True)
    #删除东道主年份
    data = data_host.drop(data_host[data_host['Host']==1].index)
    # 给缺少的年份插值
    data_host = data_host.resample(FREQ).interpolate(method='linear')
    data = data.resample(FREQ).interpolate(method='linear')
    # 法国插值至2024
    if NOC == 'France':
        data = data_host.copy(deep=True)
        data[-1:]=data[-2:-1]
    # 4年一度
    data_host.index.freq = FREQ
    return data,data_host