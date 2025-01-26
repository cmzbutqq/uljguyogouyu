import csv

path = "2025_Problem_C_Data/summerOly_medal_counts.csv"

"""
main函数返回两个字典
year_first_countries: 年份->该年第一次获奖的国家
country_first_year: 国家->该国第一次获奖的年份
ps: summerOly_medal_counts.csv里的国家都得过奖
"""


def main():
    country_first_year = {}
    year_first_countries = {}

    with open(path, "r") as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行

        for row in reader:
            noc = row[1]
            total = int(row[5])
            if total <= 0:
                continue  # 忽略无奖牌记录

            year = int(row[6])
            if noc not in country_first_year or year < country_first_year[noc]:
                country_first_year[noc] = year

    # 生成year_first_countries
    for country, year in country_first_year.items():
        if year not in year_first_countries:
            year_first_countries[year] = []
        year_first_countries[year].append(country)

    # # 按国家名称排序并输出
    # for country in sorted(country_first_year):
    #     print(f"{country},{country_first_year[country]}")

    # # 按年份排序输出
    # for year in sorted(year_first_countries):
    #     countries = sorted(year_first_countries[year])  # 国家按字母顺序排序
    #     print(f"{year}: {', '.join(countries)}")
    return year_first_countries, country_first_year


if __name__ == "__main__":
    main()
