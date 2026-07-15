import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载数据
order_product = pd.read_csv("./order_products__train.csv")
products = pd.read_csv("./products.csv")
orders = pd.read_csv("./orders.csv")
aisles = pd.read_csv("./aisles.csv")

# 2. 合并表格（构建完整数据集）
table1 = pd.merge(order_product, products, on="product_id")
table2 = pd.merge(table1, orders, on="order_id")
full_table = pd.merge(table2, aisles, on="aisle_id")

# 3. 交叉表：用户 × 商品品类（每个用户购买各类商品的次数）
user_aisle = pd.crosstab(full_table["user_id"], full_table["aisle"])

# 4. 截取前 1000 个用户进行聚类分析
data_sample = user_aisle[:1000]
print(f"降维前维度：{data_sample.shape[1]}")

# 5. PCA 降维（保留 90% 信息）
pca = PCA(n_components=0.9, random_state=22)
data_pca = pca.fit_transform(data_sample)
print(f"降维后维度: {data_pca.shape[1]}")

# 6. K-Means 聚类
kmeans = KMeans(n_clusters=5, random_state=22)
labels = kmeans.fit_predict(data_pca)

# 7. 评估（轮廓系数）
sc = silhouette_score(data_pca, labels)
print(f"轮廓系数: {sc:.3f}")

# 8. 肘部法确定最佳 K 值
sse = []
for k in range(2, 15):
    km = KMeans(n_clusters=k, random_state=22)
    km.fit(data_pca)
    sse.append(km.inertia_)
plt.figure(figsize=(8, 5))
plt.plot(range(2, 15), sse, 'bo-')
plt.xlabel('K 值')
plt.ylabel('SSE (误差平方和)')
plt.title('肘部法确定最佳 K 值')
plt.grid(True)
plt.show()

# 取轮廓系数最佳的 K
sc_list = []
for k in range(2, 11):
    labels = KMeans(n_clusters=k, random_state=22).fit_predict(data_pca)
    sc_list.append(silhouette_score(data_pca, labels))
best_k = range(2, 11)[sc_list.index(max(sc_list))]
print(f"轮廓系数法推荐的最佳 K = {best_k}")