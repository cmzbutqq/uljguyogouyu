import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import style

"""
k-means聚类分析
countries.csv
=>
country_clusters.csv
"""

SEED = 234
pd.set_option("display.max_rows", None)  # 不限制行数
# 假设df是包含上述数据的DataFrame
df = pd.read_csv("countries.csv")  # 用实际文件路径替换
# 选择要聚类的特征
features = ["CONTIN", "LNM", "CV", "Focus"]
# 标准化特征
X = StandardScaler().fit_transform(df[features])
# 确定K值
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=SEED)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


def plot_elbow_method():
    plt.figure()
    plt.plot(range(1, 11), wcss)
    plt.title("Elbow Method")
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS")
    plt.savefig("plots/kmeans/Elbow_Method.png")
    plt.show()
    plt.close("all")


plot_elbow_method()

# 选择WCSS下降速度明显变缓的点作为K值
k = 6
kmeans = KMeans(n_clusters=k, init="k-means++", random_state=SEED)
df["cluster"] = kmeans.fit_predict(X)

# 保存
df.to_csv("country_clusters.csv", index=False)


# 分析聚类结果

# 计算每个聚类的个数
cluster_counts = df[features + ["cluster"]].groupby("cluster").size()
print("Count of each cluster:")
print(cluster_counts)
# 选择数值列
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
# 使用groupby和agg方法计算每个数值列的均值和方差
cluster_stats = df.groupby("cluster")[numeric_columns].agg(["mean", "std"])
print(cluster_stats.to_string(float_format="%.2f"))


from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score


def plot2D():
    # 使用PCA进行降维
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df[features])
    # 可视化
    plt.figure()
    plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df["cluster"], cmap="viridis")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("2D KMeans Clustering after PCA")
    plt.savefig("plots/kmeans/2d.png",dpi=300)
    plt.show()
    plt.close("all")


def plot3D():
    # 使用PCA或t-SNE将数据降至3维
    pca = PCA(n_components=3)
    df_pca = pca.fit_transform(df[features])
    # 可视化
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(df_pca[:, 0], df_pca[:, 1], df_pca[:, 2], c=df["cluster"])
    plt.savefig("plots/kmeans/3d.png",dpi=300)
    plt.show()
    plt.close("all")


def plot_silhouette(data, cluster_labels, n_clusters):
    # TODO 修复这个函数
    # 计算轮廓系数
    silhouette_avg = silhouette_score(data, cluster_labels)
    sample_silhouette_values = silhouette_samples(data, cluster_labels)

    # 绘制轮廓图
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        y_lower = y_upper + 10
    plt.savefig("plots/kmeans/silhouette.png", dpi=300)

    plt.show()


import seaborn as sns

plt.figure()
sns.pairplot(df[features + ["cluster"]], hue="cluster")
plt.savefig("plots/kmeans/pairplot.png", dpi=300)
plt.show()
# plot_silhouette(df[features], df['cluster'],k)

from scipy import stats

# 对每个特征进行ANOVA分析
print("\nANOVA分析结果:")
for feature in features:
    # 按聚类分组数据
    groups = []
    for cluster in sorted(df["cluster"].unique()):
        groups.append(df[df["cluster"] == cluster][feature].values)

    # 执行ANOVA检验
    f_stat, p_value = stats.f_oneway(*groups)

    # 判断显著性（假设显著性水平为0.05）
    significant = "显著" if p_value < 0.05 else "不显著"
    print(f"{feature}: F值={f_stat:.2f}, p值={p_value:.4f} ({significant})")
