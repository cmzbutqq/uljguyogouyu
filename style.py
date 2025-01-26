import seaborn as sns
import matplotlib.pyplot as plt

# 设置 Seaborn 主题（同时影响 Matplotlib 全局参数）
sns.set_theme(
    style="whitegrid",  # 主题风格：可选 darkgrid, whitegrid, dark, white, ticks
    palette=sns.color_palette("tab20"),  # 调色板：可选 tab10, husl, Set2, deep 等
    font="sans-serif",  # 字体类型
    rc={
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "grid.color": "0.95",
    },  # 自定义参数
)


def reset():
    sns.reset_defaults()  # 恢复 Seaborn 默认参数
    plt.rcdefaults()  # 恢复 Matplotlib 默认参数


if __name__ == "__main__":
    # 绘制图表（风格统一）
    fig, ax = plt.subplots()
    sns.barplot(x=["A", "B", "C"], y=[10, 20, 15], ax=ax)
    ax.set_title("Seaborn")
    plt.show()

    fig, ax = plt.subplots()
    plt.bar(x=["A", "B", "C"], height=[10, 20, 15])
    ax.set_title("Matplotlib")
    plt.show()
