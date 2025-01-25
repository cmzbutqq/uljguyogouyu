import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro
FREQ='4YS-JAN'
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号
#读取csv
path="2025_Problem_C_Data/summerOly_medal_counts.csv"
# Rank,NOC,Gold,Silver,Bronze,Total,Year
medal_counts = pd.read_csv(path)

#选取指定国家
def get_country_data(NOC):
    data = medal_counts[(medal_counts['NOC']==NOC)&(medal_counts['Year']>=1992)]
    #年份不能重复
    assert not data['Year'].duplicated().any(),"a country with MULTIPLE records in ONE YEAR"
    # 按year排序
    data.sort_values(by='Year',inplace=True)
    # 选择年份和Total列
    data = data[['Year','Total']]
    data['Year'] = pd.to_datetime(data['Year'], format='%Y')
    data.set_index('Year', inplace=True)
    # 4年一度
    data.index.freq = '4AS-JAN'
    return data

# 事前检验：ADF 检验平稳性
def adf_test(series):
    result = adfuller(series)
    print(f'ADF 检验 p 值: {result[1]}')
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
def plot_acf_pacf(series):
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plot_acf(series, ax=plt.gca())
    plt.title('时间序列自相关图（ACF），用于分析滞后期之间的相关性')
    
    plt.subplot(122)
    plot_pacf(series, ax=plt.gca())
    plt.title('时间序列偏自相关图（PACF），用于分析去除其他滞后期影响后的相关性')
    
    return fig

# 事后检验：残差分析
def residual_analysis(model, series):
    residuals = model.resid
    resid_fig = plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plot_acf(residuals, ax=plt.gca())
    plt.title('模型残差的自相关图，检验残差的相关性结构')
    
    plt.subplot(122)
    plot_pacf(residuals, ax=plt.gca())
    plt.title('模型残差的偏自相关图，检验残差的相关性结构')
    
    

    # QQ 图
    qq_fig=qqplot(residuals, line='s')
    plt.title('残差正态性检验：QQ图，检查模型残差是否服从正态分布')

    # 正态性检验
    _, p_value = shapiro(residuals)
    print(f"Shapiro-Wilk 正态性检验 p 值: {p_value}")
    if p_value > 0.05:
        print("残差序列接近正态分布")
    else:
        print("残差序列不服从正态分布")
    
    return resid_fig,qq_fig

# 自动选择 ARIMA 模型的 p 和 q 值
def auto_select_pq(series, max_p=5, max_q=5):
    best_aic = np.inf
    best_order = None
    best_model = None

    for p in range(0, max_p+1):
        for q in range(0, max_q+1):
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
def forecast_and_plot(series, model, n_forecast, diff_series, diff_count):
    # 预测未来 n 期
    forecast = model.forecast(steps=n_forecast)
    
    # 如果数据进行了差分，需要进行多次还原
    forecast_cumsum = forecast
    if diff_count > 0:
        for i in range(diff_count):
            forecast_cumsum = forecast_cumsum.cumsum()  # 累加每次差分的结果
            forecast_cumsum += diff_series.iloc[-1]  # 加上最后一个差分值
        #差分值改成原值
        forecast_cumsum -= diff_series.iloc[-1]
        forecast_cumsum+=series.iloc[-1]
    # 绘制预测结果与实际值的折线图
    fig = plt.figure(figsize=(10, 6))
    plt.plot(series, label="实际值", color='blue')
    #type(forecast_cumsum) <class 'pandas.core.series.Series'>
    #forecast_cumsum -> idx:2028-01-01    val:123.079235 ...
    #在开头添加数据点 连上实际的折线
    forecast_cumsum = pd.concat([series[-1:], forecast_cumsum])
    plt.plot(forecast_cumsum)
    plt.title(f"ARIMA 模型预测结果与实际值比较图，展示模型的预测精度")
    plt.legend()
    return fig
# 生成模拟时间序列数据
def generate_test_series():
    np.random.seed(0)
    # 生成一个时间序列数据，包含趋势和随机噪声
    dates = pd.date_range(start='2020-01-01', periods=100, freq=FREQ)
    trend = np.linspace(10, 50, 100)  # 线性趋势
    noise = np.random.normal(0, 5, size=100)  # 随机噪声
    series = trend + noise
    data = pd.DataFrame({'Value': series}, index=dates)
    return data

# 主程序
def main(country):
    data= get_country_data(country)
    # 选择要建模的序列（例如：'Value' 列）
    series = data['Total']
    
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
    fig1=plot_acf_pacf(diff_series)
    
    # 自动选择 p, q 值
    best_model, best_order = auto_select_pq(diff_series, max_p=5, max_q=5)
    
    # 输出最优模型
    print(f"最优模型：ARIMA{best_order}")
    
    # 预测并绘制结果
    fig2=forecast_and_plot(series, best_model, n_forecast=10, diff_series=diff_series, diff_count=diff_count)
    
    # 事后检验：残差分析
    fig3,fig4=residual_analysis(best_model, series)
    
    fig1.savefig(f"plots/temp/{country}_fig1.png")
    fig2.savefig(f"plots/temp/{country}_fig2.png")
    fig3.savefig(f"plots/temp/{country}_fig3.png")
    fig4.savefig(f"plots/temp/{country}_fig4.png")

if __name__ == "__main__":
    main('United States')