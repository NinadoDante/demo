from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 生成模拟聚类数据
X, y = make_blobs(
    n_samples=300,     # 总样本数
    centers=4,         # 簇的数量
    cluster_std=0.6,   # 簇的标准差
    random_state=0
)

# 训练 K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
# kmeans.fit(x)     模型训练
# kmeans.predict(x) 模型预测
# kmeans.fit_predict(X)     # 模型训练并预测
y_pred = kmeans.fit_predict(X)

# 可视化
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=30)
plt.scatter(
    kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
    c='red', s=200, marker='X', label='质心'
)
plt.title("K-Means 聚类结果")
plt.legend()
plt.show()