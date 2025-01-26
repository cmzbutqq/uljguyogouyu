import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
提供判断某国家是否在某年举办过奥运会的函数
summerOly_medal_counts.csv
=>
medal_counts_with_host.csv
"""

# Rank,NOC,Gold,Silver,Bronze,Total,Year
medal_counts = pd.read_csv("2025_Problem_C_Data/summerOly_medal_counts.csv")
HOSTS = pd.read_csv("hosts.csv", index_col="Year")


def is_host(year, country):
    if country == "Great Britain":
        country = "United Kingdom"
    if country in HOSTS.loc[year].values:
        print(f"{country} hosted the Olympics in {year}")
        return True
    return False


medal_counts["Host"] = medal_counts.apply(
    lambda row: is_host(row["Year"], row["NOC"]), axis=1
)
print(medal_counts)
medal_counts.to_csv("medal_counts_with_host.csv")
