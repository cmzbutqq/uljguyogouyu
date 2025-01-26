import pandas as pd
import style

BEGIN_YEAR = 1992
df = pd.read_csv("2025_Problem_C_Data/summerOly_medal_counts.csv")
df = df[df["Year"] >= BEGIN_YEAR]
df = df.drop(["Rank", "Gold", "Silver", "Bronze"], axis=1)

"""
历年奖牌百分比堆积柱状图
summerOly_medal_counts.csv
=>
plots
"""


# ===数据预处理===
# 生成完整的 (country, year) 组合索引
all_countries = df["NOC"].unique()
all_years = df["Year"].unique()
full_index = pd.MultiIndex.from_product(
    [all_countries, all_years], names=["NOC", "Year"]
)

# 填充缺失值为 0
df = df.set_index(["NOC", "Year"]).reindex(full_index, fill_value=0).reset_index()

print("填充缺失值后的数据：")
print(df)  # [1435 rows x 3 columns] =>[4920 rows x 3 columns]

# ===计算百分比===
# 按 year 分组，计算每组的百分比
df["percent"] = df.groupby("Year")["Total"].transform(lambda x: x / x.sum() * 100)


# ===合并小国为 'Others'===
# 在计算百分比后添加以下代码：

# 计算每个国家的最大占比
max_percent_by_country = df.groupby("NOC")["percent"].max()

# 设定阈值
threshold = 2.0
small_countries = max_percent_by_country[max_percent_by_country < threshold].index

# 将小国合并为 "Others"
df["NOC"] = df["NOC"].where(~df["NOC"].isin(small_countries), "Others")

# 重新按 (NOC, Year) 聚合
df = df.groupby(["NOC", "Year"]).sum().reset_index()

print("计算百分比及合并后的数据：")
print(df)


# ===绘制百分比堆积柱形图===
import matplotlib.pyplot as plt
import seaborn as sns

# 初始化画布
plt.figure(figsize=(10, 6))

# 按国家顺序循环绘制堆叠条形
countries = df["NOC"].unique()
years = df["Year"].unique()

# 为 Others 单独设置灰色
main_countries = [c for c in df["NOC"].unique() if c != "Others"]
# 遍历顺序：先画 Others，避免覆盖主要国家
countries = ["Others"] + main_countries  # Others 放在最底层

palette = sns.color_palette()

# 保存每层的底部位置
bottom = pd.Series(0, index=years)

# 遍历每个国家，依次堆叠
for i, country in enumerate(countries):
    subset = df[df["NOC"] == country]
    sns.barplot(
        x="Year",
        y="percent",
        data=subset,
        color=(0.9, 0.9, 0.9) if i == 0 else palette[i % len(palette)],
        label=country,
        bottom=bottom.loc[subset["Year"]],
    )
    # 更新底部位置
    bottom += subset.set_index("Year")["percent"].reindex(years, fill_value=0)

# 添加标签和标题
plt.title(f"Percentage Stacked Bar Chart after {BEGIN_YEAR}", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Percentage (%)", fontsize=12)
plt.legend(title="Country", bbox_to_anchor=(1.05, 1), loc="upper left")

# 添加百分比标签
ax = plt.gca()
for bar in ax.containers:
    labels = [f"{h:.1f}" if h > threshold else "" for h in bar.datavalues]
    ax.bar_label(bar, labels=labels, label_type="center", fontsize=8)

plt.tight_layout()
plt.show()
