import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# CSV文件的导出 与 读取
df = pd.DataFrame({
    '班级': ['A', 'B', 'A', 'A', 'B'],
    '成绩': [85, 76, 92, 88, 69]
}, index=['h1', 'h2', 'h3', 'h4', 'h5'])
print(df)
df.to_csv('output.csv', index=False)   # 保存为 CSV 文件，包含索引
iris_df = pd.read_csv('iris.csv', encoding='utf-8')
print(iris_df.head())

# Excel（多个 Sheet）
df1 = pd.DataFrame({'ID': [1,2,3], '姓名': ['张三','李四','王五']})
df2 = pd.DataFrame({'ID': [2,3,4], '成绩': [85,90,78]})
with pd.ExcelWriter('output.xlsx') as writer:
    df1.to_excel(writer, sheet_name='Sheet1', index=False)
    df2.to_excel(writer, sheet_name='Sheet2', index=False)

# 读取 Excel 某 sheet
df1 = pd.read_excel('output.xlsx', sheet_name='Sheet1')
print(df1)

# 全局设置
plt.rcParams['font.sans-serif'] = ['SimHei']      # 显示中文
plt.rcParams['axes.unicode_minus'] = False        # 显示负号

df['成绩'].plot(kind='line', title='成绩趋势', grid=True)
plt.show()