import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# 生成一些示例数据
# 这里我们使用make_blobs生成聚类数据
from sklearn.datasets import make_blobs

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
print(X)
# 使用KMeans聚类
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# 绘制结果
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap="viridis")

# 绘制聚类中心
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c="black", s=200, alpha=0.5)
plt.title("K-means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
