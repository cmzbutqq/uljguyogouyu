import pandas as pd
import numpy as np
from scipy.stats import variation

"""
计算每个国家历年来奥运总体的奖牌得分（LNM）、获奖稳定性（CV）、参赛连续性（CONTIN）和优势项目集中度（Focus）
用于k-means分析
summerOly_athletes.csv
=>
countries.csv
"""

# 加载数据
# Name,Sex,Team,NOC,Year,City,Sport,Event,Medal
df = pd.read_csv("2025_Problem_C_Data/summerOly_athletes.csv")

# 筛除1992之前的参赛数据
df = df[df["Year"] >= 1992]
# df = df[df['Year'] ==2024]
# 删除无用列
df.drop(["Name", "Sex"], axis=1, inplace=True)
# 合并相同项，这样团体项目只有一个奖牌
df = df.drop_duplicates()

# breakpoint()

# 转换Medal列：No medal->0,_->1
df["Gold"] = df["Medal"].apply(lambda x: 1 if x in ("Gold") else 0)
df["Medal"] = df["Medal"].apply(lambda x: 1 if x in ("Bronze", "Silver", "Gold") else 0)

# df.columns ：pd.Index(['Team', 'NOC', 'Year', 'City', 'Sport', 'Event', 'Medal', 'Gold'], dtype='object')

# breakpoint()

# 按国家（NOC）分组数据
grouped = df.groupby("NOC")
# 初始化一个空的DataFrame来存储结果
results = pd.DataFrame(
    columns=[
        "NOC",
        "Country",
        "Events",
        "Medals",
        "Golds",
        "LNM",
        "CV",
        "CONTIN",
        "Focus",
    ]
)
# 遍历每个国家
for NOC, group in grouped:
    # print(NOC)
    # print(group)

    GOLDS = group["Gold"].sum()
    MEDALS = group["Medal"].sum()

    # 计算历史总奖牌得分LNM=ln(M+1)其中，M是历年（1992-2024）来得到奖牌数量总和
    LNM = np.log(MEDALS + 1)
    # 计算获奖稳定性（CV）使用变异系数CV来衡量 CV=ｓ／ｘ其中，s是标准差，ｘ是均值
    medals_per_year = group.groupby("Year")["Medal"].count()
    CV = variation(medals_per_year)
    # 计算参赛连续性（CONTIN） CONTIN=1/(k+1)，其中k,是1992年到2024年缺少参赛的届数
    years = group["Year"].unique()
    k = len(set(range(1992, 2025, 4)) - set(years))
    CONTIN = 1 / (k + 1)
    # 计算优势项目集中度（Focus）
    # 首先选出1992年到2024年得奖最多的前三个项目（大项），
    sports = group.groupby("Sport")["Medal"].sum()
    top_sports = sports.nlargest(3).sum()
    # if NOC == 'USA':
    #     breakpoint()

    # 三个项目的奖牌数量之和与总奖牌数量之比,若总奖牌数量为零或者十年来参与项目不足三次，记为0
    if MEDALS == 0 or len(group) < 3:
        Focus = 0
    else:
        Focus = top_sports / MEDALS
    # 将结果添加到DataFrame
    new_record = pd.DataFrame(
        [
            {
                "NOC": NOC,
                "Country": group["Team"].unique()[0],
                "Events": len(group),
                "Medals": MEDALS,
                "Golds": GOLDS,
                "LNM": LNM,
                "CV": CV,
                "CONTIN": CONTIN,
                "Focus": Focus,
            }
        ]
    )
    results = pd.concat([results, new_record], ignore_index=True)
results = results.sort_values(by="Medals", ascending=False)
print(results)
# 保存结果
results.to_csv("countries.csv", index=False)
