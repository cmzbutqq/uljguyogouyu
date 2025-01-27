import pandas as pd
import json

"""
2025_Problem_C_Data/summerOly_athletes.csv是记录了历届奥运会各国运动员获奖情况的csv长表格。
表头：Name,Sex,Team,NOC,Year,City,Sport,Event,Medal
内容举例：Minna Aalto,F,Finland,FIN,2000,Sydney,Sailing,Sailing Women's Windsurfer,No medal
Medal有四种类型:No medal,Gold,Silver,Bronze

先删除Name,Sex两列，再drop_duplicate，这样团体项目只记一次奖牌。

你现在需统计对于新增的项目(Event)，获取项目在增加后，那一届和后面两届的获奖国家，保存为json格式。
内容举例：[
{
"event":"Gymnastics Men's Individual All-Around" ,
"add_year":1952,
0:["country1","country2"...], #项目新增时的那一届 这个例子里就是1952年这个event获奖的国家
1:["country1","country2"...], #项目新增后的那一届 1956年这个event获奖的国家
2:["country1","country2"...] #项目新增后的第二届 1960年这个event获奖的国家
},
...

]

"""
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
