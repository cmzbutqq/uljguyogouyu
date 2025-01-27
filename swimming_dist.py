import pandas as pd
import matplotlib.pyplot as plt

# 读取数据并预处理
df = pd.read_csv("2025_Problem_C_Data/summerOly_athletes.csv")
df = df.drop(columns=["Name", "Sex"]).drop_duplicates()

# 筛选1992年后、运动项目包含"Swimming"、且获奖的数据
df = df[
    (df["Year"] >= 1992)
    & (df["Sport"].str.contains("Swimming"))  # 关键修改：筛选所有游泳相关项目
    & (df["Medal"] != "No medal")
]

# 统计所有国家奖牌总数
medal_counts = df["Team"].value_counts().reset_index()
medal_counts.columns = ["Country", "Total Medals"]
medal_counts = medal_counts.sort_values("Total Medals", ascending=False)

# 计算变异系数CV（标准差/均值，百分比形式）
cv = (medal_counts["Total Medals"].std() / medal_counts["Total Medals"].mean()) * 100
print(f"CV of Swimming: {cv:.2f}%")

# 提取前10名用于饼图
top10 = medal_counts.head(10)
others_sum = medal_counts["Total Medals"][10:].sum()
pie_data = pd.concat(
    [top10, pd.DataFrame({"Country": ["Others"], "Total Medals": [others_sum]})]
)

# 设置全局样式
plt.style.use("ggplot")

# --------------------------
# 绘制所有国家的柱状图（纵向，标签可能拥挤）
# --------------------------
plt.figure(figsize=(15, 8))
bars = plt.bar(
    medal_counts["Country"], medal_counts["Total Medals"], color="#2ca02c"
)  # 修改颜色区分项目
plt.xlabel("Country", fontsize=12)
plt.ylabel("Total Medals", fontsize=12)
plt.title("All Countries Swimming Medals (1992-2020)", fontsize=14)
plt.xticks(rotation=90, ha="center", fontsize=8)
plt.tight_layout()
plt.savefig("plots/swimming_dist_bar.png", dpi=300)
plt.show()


# --------------------------
# 绘制前10国家+Others的饼图
# --------------------------
plt.figure(figsize=(10, 10))
colors = plt.cm.Paired.colors  # 使用更鲜明的色系
wedges, texts, autotexts = plt.pie(
    pie_data["Total Medals"],
    labels=pie_data["Country"],
    autopct="%1.1f%%",
    startangle=140,
    colors=colors,
    textprops={"fontsize": 10},
)
plt.setp(autotexts, size=10, weight="bold")
plt.title("Top 10 Countries Swimming Medals (1992-2020)", fontsize=14)
plt.axis("equal")
plt.tight_layout()
plt.savefig("plots/swimming_dist_pie.png", dpi=300)
plt.show()
