import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

folder = "2025_Problem_C_Data/"


def load(file_name, encoding="utf8"):
    return pd.read_csv(folder + file_name, encoding=encoding)


# Year,Host
hosts = load("summerOly_hosts.csv")
# Rank,NOC,Gold,Silver,Bronze,Total,Year
medals = load("summerOly_medal_counts.csv")
# Name,Sex,Team,NOC,Year,City,Sport,Event,Medal
athletes = load("summerOly_athletes.csv")
# Sport,Discipline,Code,Sports Governing Body,1896,1900,1904,...,2024
programs = load("washed_summerOly_programs.csv", encoding="windows-1252")

y = "Sport"
df, name = (programs, "programs")
print(f"unique {y} values of {name} are:")
print(df[y].unique().shape)

df, name = (athletes, "athletes")
print(f"unique {y} values of {name} are:")
print(df[y].unique().shape, df[y].unique())

y = "Event"
df, name = (athletes, "athletes")
print(f"unique {y} values of {name} are:")
print(df[y].unique().shape)

y = "Discipline"
df, name = (programs, "programs")
print(f"unique {y} values of {name} are:")
print(df[y].unique().shape)

# dataframes = hosts, medals, programs
# [print(df.head(), "\n", df.shape) for df in dataframes]

# # 计算占比
# for year in range(1896, 2024 + 1, 4):
#     filt = medals["Year"] == year
#     this_year = medals.loc[filt, :]
#     total_medals = this_year["Total"].sum()
#     medals.loc[filt, "Percentage"] = this_year["Total"] / total_medals


# plt.figure()
# # 过滤
# filt = (medals["Year"] > 1961) & (medals["Rank"] <= 5)
# medals = medals.loc[filt, :]
# countrys = medals["NOC"].unique()
# sns.lineplot(x="Year", y="Percentage", data=medals, hue="NOC")
# sns.scatterplot(x="Year", y="Percentage", data=medals, hue="NOC")
# plt.show()
