from sklearn.linear_model import Ridge, Lasso, ElasticNet
import numpy as np

np.random.seed(42)
X = np.random.rand(100, 10)
y = X.dot([1,2,3,0,0,0,0,0,0,0]) + np.random.normal(0, 0.1, 100)

ridge = Ridge(alpha=1.0).fit(X, y)
lasso = Lasso(alpha=0.1).fit(X, y)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X, y)

print("Ridge 系数:", ridge.coef_)
print("Lasso 系数:", lasso.coef_)
print("ElasticNet 系数:", elastic.coef_)


# ... existing code ...

print("真实系数:  ", [1, 2, 3, 0, 0, 0, 0, 0, 0, 0])
print("Ridge 系数:", np.round(ridge.coef_, 4))
print("Lasso 系数:", np.round(lasso.coef_, 4))
print("ElasticNet 系数:", np.round(elastic.coef_, 4))
