import joblib

import numpy as np
import matplotlib.pyplot as plt

X = np.array([1, 2, 3, 4, 5])
y = np.array([2.2, 3.9, 5.8, 7.9, 10.1])

# 正规方程解
X_b = np.c_[X, np.ones_like(X)]  # 添加偏置列
w = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
print(f"系数 w: {w[0]:.2f}, 偏置 b: {w[1]:.2f}")

plt.scatter(X, y)
plt.plot(X, X_b @ w, 'r-')
plt.show()



