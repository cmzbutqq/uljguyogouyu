import pandas as pd

file_path = "2025_Problem_C_Data/summerOly_athletes.csv"
df = pd.read_csv(file_path)

all_events = df["Event"].unique()
all_noc = df["NOC"].unique()


def get_all_events_first_medal(noc_code):
    """
    返回指定国家在所有 Event 中的首次获奖年份（包含从未获奖的 Event）。
    输出格式：国家（NOC）、Event、年份（若无则标记为 "从未获得"）。
    """
    # 读取数据并获取所有唯一的 Event 列表

    # 过滤该国的有效奖牌记录
    country_medals = df[(df["NOC"] == noc_code) & ~(df["Medal"] == "No medal")]

    # 按 Event 分组并找到最小年份
    event_min_year = country_medals.groupby("Event")["Year"].min()

    # 构建结果 DataFrame（覆盖所有 Event）
    result = pd.DataFrame({"Event": all_events})
    result["NOC"] = noc_code

    # 合并首次获奖年份数据
    result = result.merge(
        event_min_year.rename("First_Year"),
        how="left",
        left_on="Event",
        right_index=True,
    )

    # 标记未获奖的 Event
    result["First_Year"] = result["First_Year"].fillna("Never")

    return result[["NOC", "Event", "First_Year"]]


def get_all_countries_first_medal(
    output_csv_path="countries_events_first_medal_year.csv",
):
    """
    将所有国家（NOC）在每个 Event 中的首次获奖年份汇总到一个CSV文件
    格式：NOC（国家）, Event, First_Year（年份或"Never"）
    """
    # 存储所有结果的容器
    final_results = []
    # 遍历每个国家
    for noc in all_noc:
        # 获取该国所有 Event 的首次获奖年份
        all_events = get_all_events_first_medal(noc)

        # 添加到结果列表
        final_results.append(all_events)

    # 合并所有结果并保存
    pd.concat(final_results).to_csv(output_csv_path, index=False)
    print(f"结果已保存至 {output_csv_path}")


if __name__ == "__main__":
    get_all_countries_first_medal()
