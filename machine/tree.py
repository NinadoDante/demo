import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz

# 1. 加载数据
titan = pd.read_csv("./train.csv")

# 2. 选取特征与目标
x = titan[["Pclass", "Age", "Sex"]].copy()
y = titan["Survived"]

# 3. 缺失值处理
x["Age"] = x["Age"].fillna(x["Age"].mean())

# 4. 划分数据集
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=22, stratify=y
)

# 5. 特征工程：字典向量化（类别特征转为独热编码）
transfer = DictVectorizer(sparse=False)
x_train = transfer.fit_transform(x_train.to_dict(orient="records"))
x_test = transfer.transform(x_test.to_dict(orient="records"))

# 6. 训练决策树（预剪枝）
model = DecisionTreeClassifier(
    criterion="entropy",   # 使用信息熵，也可选 "gini"
    max_depth=5,
    min_samples_leaf=5,
    random_state=22
)
model.fit(x_train, y_train)

# 7. 评估
accuracy = model.score(x_test, y_test)
print(f"测试准确率: {accuracy:.2%}")

# 8. 决策树可视化
export_graphviz(
    model,
    out_file="./tree.dot",
    feature_names=transfer.get_feature_names_out(),
    class_names=["遇难", "幸存"],
    filled=True,
    rounded=True
)
# 生成的 dot 文件可复制到 http://webgraphviz.com/ 查看图形