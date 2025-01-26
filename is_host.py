import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#读取csv
path="2025_Problem_C_Data/summerOly_medal_counts.csv"
# Rank,NOC,Gold,Silver,Bronze,Total,Year
MEDAL_COUNTS = pd.read_csv(path)
MEDAL_COUNTS=MEDAL_COUNTS[MEDAL_COUNTS['Year']>=BEGIN_YEAR]