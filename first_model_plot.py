import csv
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号


def main():
    """
    处理夏季奥运会奖牌数据，返回：
    - year_first_countries: 年份为键，首次获奖国家列表为值
    - country_first_year: 国家为键，首次获奖年份为值
    """
    # 初始化数据结构
    country_first_year = {}
    year_first_countries = {}

    try:
        with open(
            "2025_Problem_C_Data/summerOly_medal_counts.csv", "r", encoding="utf-8"
        ) as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)  # 读取并验证标题行

            # 确保CSV格式正确
            if (
                len(header) < 7
                or header[1] != "NOC"
                or header[5] != "Total"
                or header[6] != "Year"
            ):
                raise ValueError("CSV文件格式不符合预期")

            for row in csv_reader:
                # 数据有效性检查
                if len(row) < 7:
                    continue

                noc = row[1].strip()
                if not noc:  # 跳过空国家代码
                    continue

                try:
                    total = int(row[5])
                    year = int(row[6])
                except (ValueError, IndexError):
                    continue  # 跳过无效数据行

                if total <= 0:
                    continue  # 忽略无奖牌记录

                # 更新首次获奖年份
                if noc not in country_first_year or year < country_first_year[noc]:
                    country_first_year[noc] = year

    except FileNotFoundError:
        print(f"错误：文件未找到，请检查路径是否正确")
        return {}, {}
    except Exception as e:
        print(f"数据处理发生错误：{str(e)}")
        return {}, {}

    # 构建年份->国家字典
    for country, year in country_first_year.items():
        year_first_countries.setdefault(year, []).append(country)

    # 按年份排序输出
    print("各年份首次获奖国家列表：")
    for year in sorted(year_first_countries.keys()):
        countries = sorted(year_first_countries[year])
        print(f"{year}年 ({len(countries)}国): {', '.join(countries)}")

    return year_first_countries, country_first_year


def visualize_first_wins(year_data):
    """可视化首奖国家年度分布"""
    if not year_data:
        print("无有效数据可供可视化")
        return

    plt.figure(figsize=(14, 7))
    years = sorted(year_data.keys())
    counts = [len(year_data[y]) for y in years]

    # 创建柱状图
    bars = plt.bar(years, counts, color="steelblue", width=2.5)

    # 标记特殊年份
    highlights = {
        1896: "first Olympics",
        1948: "after WWII",
        1960: "year of Africa",
        1992: "Dissolution of the Soviet Union",
        1996: "",
    }
    for bar in bars:
        year = bar.get_x() + bar.get_width() / 2
        if year in highlights:
            bar.set_color("indianred")
            plt.text(
                year,
                bar.get_height() + 0.5,
                highlights[year],
                ha="center",
                va="bottom",
                rotation=45,
                fontsize=10,
                color="darkred",
                fontstyle="italic",
            )

    # 图表设置
    plt.title(
        "Distribution of first Summer Olympics medals by country (1896-2012)",
        fontsize=14,
        pad=20,
    )
    plt.xlabel("Year", fontsize=12, labelpad=10)
    plt.ylabel(
        "Number of countries receiving their first medals",
        fontsize=12,
        labelpad=10,
    )
    plt.xticks(years, rotation=60, fontsize=9)
    plt.yticks(fontsize=9)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # 添加数值标签
    for year, count in zip(years, counts):
        plt.text(year, count + 0.3, str(count), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig("plots/first_medal/bar.png",dpi=300)
    plt.show()


if __name__ == "__main__":
    # 执行主程序
    year_dict, country_dict = main()

    # 显示基本统计信息
    if year_dict and country_dict:
        print(f"\n统计摘要：")
        print(f"总国家数：{len(country_dict)}")
        print(f"时间跨度：{min(year_dict.keys())}-{max(year_dict.keys())}")
        print(f"最多新增年份：{max(year_dict, key=lambda y: len(year_dict[y]))}年")

        # 生成可视化
        visualize_first_wins(year_dict)
