from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# 1. 获取数据
iris = load_iris()
X, y = iris.data, iris.target
print(f"===== 第1步：获取数据 =====")
print(f"特征数据形状: {X.shape}")
print(f"标签数据形状: {y.shape}")
print(f"特征名称: {iris.feature_names}")
print(f"目标类别: {iris.target_names}")
print(f"前5条特征数据:\n{X[:5]}")
print(f"前5条标签数据: {y[:5]}")

# 2. 划分训练集和测试集 比例 8:2
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=22
)
print(f"\n===== 第2步：划分训练集和测试集 =====")
print(f"训练集特征形状: {X_train.shape}")
print(f"测试集特征形状: {X_test.shape}")
print(f"训练集标签形状: {y_train.shape}")
print(f"测试集标签形状: {y_test.shape}")

# 3. 【关键步骤】特征预处理：标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(f"\n===== 第3步：特征标准化 =====")
print(f"标准化后训练集前3条数据:\n{X_train[:3]}")
print(f"标准化后各特征均值: {X_train.mean(axis=0).round(4)}")
print(f"标准化后各特征标准差: {X_train.std(axis=0).round(4)}")

# 4. 模型训练 + 网格搜索（自动找最佳K值）
param_grid = {"n_neighbors": [1, 3, 5, 7, 9]}
model = KNeighborsClassifier()
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(f"\n===== 第4步：模型训练 + 网格搜索 =====")
print(f"搜索参数: {param_grid}")
print(f"交叉验证折数: 5")
print(f"各参数交叉验证得分: {grid_search.cv_results_['mean_test_score']}")
print(f"最佳K值: {grid_search.best_params_['n_neighbors']}")
print(f"交叉验证最高得分: {grid_search.best_score_:.2%}")

# 5. 模型评估
best_model = grid_search.best_estimator_
accuracy = best_model.score(X_test, y_test)
print(f"\n===== 第5步：模型评估 =====")
print(f"最佳模型参数: {best_model.get_params()}")
print(f"测试准确率: {accuracy:.2%}")

# 6. 用最佳模型预测一个全新的样本
sample = [[5.1, 3.5, 1.4, 0.2]]
sample_scaled = scaler.transform(sample)
prediction = best_model.predict(sample_scaled)
print(f"\n===== 第6步：新样本预测 =====")
print(f"输入样本: {sample[0]}")
print(f"标准化后: {sample_scaled[0].round(4)}")
print(f"预测品种: {iris.target_names[prediction[0]]}")