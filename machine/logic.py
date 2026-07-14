import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

# 1. 加载数据
# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
           'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']
df = pd.read_csv("pima-indians-diabetes.csv", names=columns)

# 2. 处理异常值
for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())  # 去掉 inplace=True，直接赋值

X = df.iloc[:, :-1]
y = df['Outcome']
print("数据集样本数:", df.shape[0])

# 3. 划分 + 标准化
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. 逻辑回归 + 网格搜索
model = LogisticRegression(solver='saga', max_iter=200)
param_grid = {'C': [0.01, 0.1, 1, 10], 'l1_ratio': [0, 1]}
grid = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
grid.fit(X_train, y_train)

# 5. 评估
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("最佳参数:", grid.best_params_)
print("AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred, target_names=['非糖尿病', '糖尿病']))

# 6. 绘制 ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc_score(y_test, y_prob):.2f})")
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC 曲线')
plt.legend()
plt.grid(True)
plt.show()