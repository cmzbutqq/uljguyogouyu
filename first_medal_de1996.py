import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号

path = "2025_Problem_C_Data/summerOly_medal_counts.csv"


def main():
    """主数据处理函数"""
    country_first_year = {}
    year_first_countries = {}

    with open(path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        headers = next(reader)  # 读取标题行

        # 验证CSV格式
        if (
            len(headers) < 7
            or headers[1] != "NOC"
            or headers[5] != "Total"
            or headers[6] != "Year"
        ):
            raise ValueError("CSV文件格式不符合预期")

        for row in reader:
            if len(row) < 7:
                continue  # 跳过不完整行

            noc = row[1].strip()
            try:
                total = int(row[5])
                year = int(row[6])
            except (ValueError, IndexError):
                continue  # 跳过无效数据

            if total > 0:  # 仅处理有奖牌记录
                # 更新首次获奖年份
                if noc not in country_first_year or year < country_first_year[noc]:
                    country_first_year[noc] = year

    # 构建年份字典
    for country, year in country_first_year.items():
        year_first_countries.setdefault(year, []).append(country)

    # 按年份排序输出
    print("各年份首次获奖国家：")
    for year in sorted(year_first_countries):
        countries = sorted(year_first_countries[year])
        print(f"{year}: {len(countries)}国 - {', '.join(countries)}")

    return year_first_countries, country_first_year


def normality_analysis(year_data):
    """正态性检验分析"""
    # 筛选样本：1992-2024年，排除1996
    valid_years = [y for y in year_data if 1992 <= y <= 2024 and y != 1996]
    sample = [len(year_data[y]) for y in sorted(valid_years)]

    # 显示样本信息
    print("\n=== 分析样本 ===")
    print(f"时间范围：{min(valid_years)}-{max(valid_years)}")
    print(f"排除年份：1996")
    print(f"样本数量：{len(sample)}")
    print("年份分布：")
    for y in sorted(valid_years):
        print(f"  {y}: {len(year_data[y])}国")
        
    
    # 新增置信区间计算（在样本信息输出部分之后）
    print("\n=== 置信区间计算 ===")
    confidence_level = 0.95
    n = len(sample)
    mean = np.mean(sample)
    std_err = stats.sem(sample)  # 标准误

    # 使用t分布计算置信区间（适用于小样本）
    ci_low, ci_high = stats.t.interval(
        confidence=confidence_level, df=n - 1, loc=mean, scale=std_err
    )
    print(
        f"均值置信区间（{int(confidence_level*100)}%）: ({ci_low:.2f}, {ci_high:.2f})"
    )
    
    
    # 执行正态性检验
    print("\n=== 正态性检验结果 ===")
    # Shapiro-Wilk检验（适合小样本）
    shapiro_stat, shapiro_p = stats.shapiro(sample)
    print(f"Shapiro-Wilk检验：W = {shapiro_stat:.3f}, p = {shapiro_p:.4f}")

    # Kolmogorov-Smirnov检验
    ks_stat, ks_p = stats.kstest(sample, "norm", args=(np.mean(sample), np.std(sample)))
    print(f"Kolmogorov-Smirnov检验：D = {ks_stat:.3f}, p = {ks_p:.4f}")

    # 可视化诊断图
    plt.figure(figsize=(12, 5))

    # 直方图
    plt.subplot(1, 2, 1)
    # sns.histplot(sample, kde=True, stat="density", color="steelblue", element="step")
    sns.kdeplot(data=sample, color="steelblue")
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, np.mean(sample), np.std(sample))
    plt.plot(x, p, "r--", linewidth=2)
    plt.title("PDF Histogram")
    plt.xlabel("Number of additional countries")
    plt.ylabel("Density of Probability")

    # Q-Q图
    plt.subplot(1, 2, 2)
    stats.probplot(sample, dist="norm", plot=plt)
    plt.title("Q-Q Plot")

    plt.tight_layout()
    plt.savefig("plots/first_medal/normality_analysis.png", dpi=300)
    plt.show()

    # 结果解读
    alpha = 0.05
    conclusion = "服从正态分布" if shapiro_p > alpha else "不服从正态分布"
    print(f"\n结论（α={alpha}）：数据{conclusion}")
    return x, p


if __name__ == "__main__":
    # 执行主程序
    year_dict, country_dict = main()

    # 显示基本统计
    print("\n=== 全局统计 ===")
    print(f"总国家数：{len(country_dict)}")
    print(f"时间跨度：{min(year_dict)}-{max(year_dict)}")

    # 执行正态性检验
    if year_dict:
        normality_analysis(year_dict)
