import pandas as pd

"""
2025_Problem_C_Data/summerOly_athletes.csv是记录了历届奥运会各国运动员获奖情况的csv长表格。
表头：Name,Sex,Team,NOC,Year,City,Sport,Event,Medal
内容举例：Minna Aalto,F,Finland,FIN,2000,Sydney,Sailing,Sailing Women's Windsurfer,No medal
Medal有四种类型:No medal,Gold,Silver,Bronze


你现在需统计哪些运动（Sport）对哪些国家（NOC）更加重要，生成一个新的长表格
表头 : NOC,Sport,type1,type2,type3,importance
(1)该国家该项目奖牌 占据了 该项目所有国家总奖牌数 的大部分(threshold1=30%)，认为是第一类特色项目（type1=True）
(2)该国家该项目 连续获奖超过三届（四年一届：2000,2004,2008这种，不用管停办的奥运会），认为是第二类特色项目（type2=True）
(3)该国家该项目奖牌 占据了 该国家所有项目总奖牌数 的大部分(threshold3=30%)，认为是第三类特色项目（type3=True）
如果一个项目符合三种特色项目的标准，认为这个项目对这个国家特别重要（importance=3）
如果一个项目符合两种特色项目的标准，认为这个项目对这个国家比较重要（importance=2）
如果一个项目符合一种特色项目的标准，认为这个项目对这个国家一般重要（importance=1）
"""


# 读取数据并预处理
df = pd.read_csv("2025_Problem_C_Data/summerOly_athletes.csv")
# 先删除Name,Sex两列，再drop_duplicate，这样团体项目只记一次奖牌。
df = df.drop(columns=["Name", "Sex"])
df = df.drop_duplicates()
df = df[df["Medal"] != "No medal"]  # 只保留获奖记录

# 条件1：该国家该项目奖牌 占据了 该项目所有国家总奖牌数 >=10%
sport_total = df.groupby("Sport")["Medal"].count().reset_index(name="sport_total")
noc_sport_count = (
    df.groupby(["NOC", "Sport"])["Medal"].count().reset_index(name="noc_sport_count")
)
merged = noc_sport_count.merge(sport_total, on="Sport", how="left")
merged["type1"] = merged["noc_sport_count"] / merged["sport_total"] >= 0.1


# 条件2：连续三届获奖（i, i+4, i+8）
def check_consecutive(years):
    years_set = set(years)
    for y in years_set:
        if (y + 4 in years_set) and (y + 8 in years_set):
            return True
    return False


noc_sport_years = df.groupby(["NOC", "Sport"])["Year"].apply(list).reset_index()
noc_sport_years["type2"] = noc_sport_years["Year"].apply(check_consecutive)
merged = merged.merge(
    noc_sport_years[["NOC", "Sport", "type2"]], on=["NOC", "Sport"], how="left"
)

# 条件3：该国家该项目奖牌 占据了 该国家所有项目总奖牌数 >=30%
noc_total = df.groupby("NOC")["Medal"].count().reset_index(name="noc_total")
merged = merged.merge(noc_total, on="NOC", how="left")
merged["type3"] = merged["noc_sport_count"] / merged["noc_total"] >= 0.3

# 计算重要性并过滤
merged["importance"] = merged[["type1", "type2", "type3"]].sum(axis=1)
result = merged[["NOC", "Sport", "type1", "type2", "type3", "importance"]]
result = result[result["importance"] > 0]  # 排除importance=0

# 输出结果
result.to_csv("sport_noc_importance.csv", index=False)
