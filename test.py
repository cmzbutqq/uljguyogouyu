import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np

folder = "2025_Problem_C_Data/"


def load(file_name, encoding="utf8"):
    return pd.read_csv(folder + file_name, encoding=encoding)

# 运动员~年份~国家~项目~奖牌
athletes = load("summerOly_athletes.csv")
# 举办者
hosts = load("summerOly_hosts.csv")
# 国家~奖牌数~年份
medals = load("summerOly_medal_counts.csv")
# 项目~各年次数
programs = load("summerOly_programs.csv",encoding="windows-1252")

dataframes = athletes, hosts, medals, programs
[print(df.head(), "\n", df.shape) for df in dataframes]
