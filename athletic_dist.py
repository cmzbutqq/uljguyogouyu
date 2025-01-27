import pandas as pd
import matplotlib.pyplot as plt

# 读取数据并预处理
df = pd.read_csv("2025_Problem_C_Data/summerOly_athletes.csv")
df = df.drop(columns=["Name", "Sex"]).drop_duplicates()
df = df[
    (df["Year"] >= 1992) & (df["Sport"] == "Athletics") & (df["Medal"] != "No medal")
]

# 统计所有国家奖牌总数（已按降序排列）
medal_counts = df["Team"].value_counts().reset_index()
medal_counts.columns = ["Country", "Total Medals"]
medal_counts = medal_counts.sort_values("Total Medals", ascending=False)

# 提取前10名用于饼图（含Others合并）
top10 = medal_counts.head(10)
others_sum = medal_counts["Total Medals"][10:].sum()
pie_data = pd.concat(
    [top10, pd.DataFrame({"Country": ["Others"], "Total Medals": [others_sum]})]
)

# 设置全局样式
plt.style.use("ggplot")

# --------------------------
# 绘制所有国家的柱状图（完整排名）
# --------------------------
plt.figure(figsize=(15, 8))
bars = plt.bar(medal_counts["Country"], medal_counts["Total Medals"], color="#1f77b4")
plt.xlabel("Country", fontsize=12)
plt.ylabel("Total Medals", fontsize=12)
plt.title("All Countries Athletics Medals (1992-2020)", fontsize=14)
plt.xticks(rotation=90, ha="center", fontsize=8)  # 垂直旋转标签
plt.tight_layout()

# 可选：添加数据标签（因数量多可能重叠，谨慎使用）
# for bar in bars:
#     height = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2., height, f'{height}',
#              ha='center', va='bottom', fontsize=6, rotation=90)

plt.savefig("plots/athletic_dist_bar.png", dpi=300)
plt.show()

# --------------------------
# 绘制前10国家+Others的饼图
# --------------------------
plt.figure(figsize=(10, 10))
colors = plt.cm.tab20c.colors
wedges, texts, autotexts = plt.pie(
    pie_data["Total Medals"],
    labels=pie_data["Country"],
    autopct="%1.1f%%",
    startangle=140,
    colors=colors,
    textprops={"fontsize": 10},
)
plt.setp(autotexts, size=10, weight="bold")
plt.title("Top 10 Countries Athletics Medals (1992-2020)", fontsize=14)
plt.axis("equal")
plt.tight_layout()
plt.savefig("plots/athletic_dist_pie.png", dpi=300)
plt.show()
