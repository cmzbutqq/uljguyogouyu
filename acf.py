import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.tsa.stattools import acf
from itertools import product

medals = pd.read_csv("2025_Problem_C_Data/" + "summerOly_medal_counts.csv")
print(medals.head())
COUNTRIES = (
    "China",
    "Australia",
    "Japan",
    "Great Britain",
    "United States",
    "France",
    "Canada",
    "Germany",
    "Kenya",
    "Jamaica",
)
YEARS=1992,1896
Ys="Gold","Total"
for start_year,y,country in product(YEARS,Ys,COUNTRIES):
    country_data = medals[(medals["NOC"] == country) & (medals["Year"] >= start_year)]
    acf_values = acf(country_data[y].values)
    plt.figure()
    plt.stem(acf_values)
    plt.title(f"ACF for {country} in {y} medals from {start_year}")
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.savefig(f"plots/acf/ACF_{country}_{y}_{start_year}.png")
