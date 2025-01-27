import pandas as pd
import json

# 读取数据并预处理
df = pd.read_csv("2025_Problem_C_Data/summerOly_athletes.csv")
df = df.drop(columns=["Name", "Sex"])
df = df.drop_duplicates()
df = df[df["Medal"] != "No medal"]

# 确定每个Event的add_year（首次出现的年份）
event_add_year = df.groupby("Event")["Year"].min().reset_index()
event_add_year.columns = ["Event", "add_year"]

# 创建每个Event在每个Year的国家列表
event_year_countries = df.groupby(["Event", "Year"])["Team"].unique().reset_index()
event_year_countries["Team"] = event_year_countries["Team"].apply(lambda x: list(x))

# 构建结果数据结构
result = []
for _, row in event_add_year.iterrows():
    event = row["Event"]
    add_year = row["add_year"]
    entry = {"event": event, "add_year": int(add_year), 0: [], 1: [], 2: []}
    # 检查三个目标年份：add_year, add_year+4, add_year+8
    target_years = [add_year, add_year + 4, add_year + 8]
    for idx, target_year in enumerate(target_years):
        # 查找该Event和target_year的记录
        mask = (event_year_countries["Event"] == event) & (
            event_year_countries["Year"] == target_year
        )
        filtered = event_year_countries[mask]
        if not filtered.empty:
            countries = filtered.iloc[0]["Team"]
            entry[idx] = sorted(countries)  # 按字母顺序排序，可选
        else:
            entry[idx] = []
    result.append(entry)

# 保存为JSON文件
with open("new_events_medal_countries.json", "w") as f:
    json.dump(result, f, indent=2)
