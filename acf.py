import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.tsa.stattools import acf

medals = pd.read_csv("2025_Problem_C_Data/" + "summerOly_medal_counts.csv")
print(medals.head())
countries = (
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
START_YEAR=1896
Y_VAL="Total"
for country in countries:
    country_data = medals[(medals["NOC"] == country) & (medals["Year"] >= START_YEAR)]
    acf_values = acf(country_data[Y_VAL].values)
    plt.figure()
    plt.stem(acf_values)
    plt.title(f"ACF for {country}")
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.savefig(f"plots/acf/ACF_{country}_{Y_VAL}_{START_YEAR}.png")
