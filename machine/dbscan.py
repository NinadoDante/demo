from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# 生成半月形数据（非球形、含噪声）
X, _ = make_moons(n_samples=200, noise=0.05, random_state=0)

# DBSCAN 聚类
dbscan = DBSCAN(eps=0.3, min_samples=5)
y_pred = dbscan.fit_predict(X)   # 噪声点标签为 -1

# 可视化
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis')
plt.title("DBSCAN 聚类结果（标签为 -1 的为噪声点）")
plt.show()

print(f"聚类标签（-1=噪声）: {set(y_pred)}")
print(f"噪声点数量: {(y_pred == -1).sum()}")