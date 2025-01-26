import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import first_medal

"""
2024年参加过比赛 但从未得到过奖牌的国家(Team)
！！！这里面的队伍有很多是地区和重名，本质是队伍而非国家，注意鉴别
"""

ATHLETES = pd.read_csv("2025_Problem_C_Data/summerOly_athletes.csv")
ATHLETES = ATHLETES[ATHLETES["Year"] == 2024]
countries = ATHLETES["Team"].unique()
all_countries = set()
for c in countries:
    all_countries.add(c)
y2c, c2y = first_medal.main()
medaled = set(c2y.keys())

print(sorted(all_countries - medaled))  # unmedaled
