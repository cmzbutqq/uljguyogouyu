import json
import csv
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

"""
计算1992年之后每个国家在新项目(新增的当年)获得了多少奖牌，哪些奖牌
"""


def make_csv():
    # 读取JSON数据
    with open("new_events_medal_countries.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # 初始化国家信息存储结构
    country_info = defaultdict(lambda: {"medals": 0, "events": []})

    # 处理每个事件条目
    for entry in data:
        add_year = entry.get("add_year", 0)
        if add_year <= 1992:
            continue  # 忽略1992年及之前的新增项目

        event_name = entry.get("event", "")
        current_year_countries = entry.get("0", [])

        for country in current_year_countries:
            country_info[country]["medals"] += 1
            country_info[country]["events"].append(event_name)

    # 准备CSV内容并按国家名称排序
    csv_rows = []
    for country in sorted(country_info.keys()):
        medals = country_info[country]["medals"]
        events = country_info[country]["events"]
        events_str = ", ".join(events)
        csv_rows.append([country, medals, events_str])

    # 写入CSV文件
    with open("country_new_medals.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Team", "new_event_medal", "events"])
        writer.writerows(csv_rows)


df = pd.read_csv("country_new_medals.csv")


def bar_rank():

    # 读取数据

    # 按奖牌数排序并绘图
    df_sorted = df.sort_values("new_event_medal", ascending=False)[:50]
    # 取前50位
    plt.figure(figsize=(12, 6))
    plt.bar(df_sorted["Team"], df_sorted["new_event_medal"], color="skyblue")
    plt.title("New Event Medal Count by Country (Post-1992)")
    plt.xlabel("Country")
    plt.ylabel("Number of Medals")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("plots/country_new_medals.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    make_csv()
    bar_rank()
