from sklearn.neighbors import KNeighborsClassifier

# 1. 准备数据（特征 X 和 标签 y）
# 特征：搞笑镜头, 拥抱镜头, 打斗镜头
X = [
    [39, 0, 31], [3, 2, 65], [2, 3, 55], [9, 38, 2],
    [8, 34, 17], [5, 2, 57], [21, 17, 5], [45, 2, 9]
]
y = ['喜剧片', '动作片', '爱情片', '爱情片',
     '爱情片', '动作片', '喜剧片', '喜剧片']

# 2. 创建模型并训练
model = KNeighborsClassifier(n_neighbors=3)  # K值取3
model.fit(X, y)

# 3. 预测新电影《唐人街探案》
test = [[23, 3, 17]]  # 23个搞笑镜头，3个拥抱镜头，17个打斗镜头
print(f"预测类型: {model.predict(test)[0]}")  # 输出：喜剧片

# 4. 评估准确率
accuracy = model.score(X, y)  # 在训练集上评估（实际应用应在测试集上）
print(f"准确率: {accuracy:.2%}")