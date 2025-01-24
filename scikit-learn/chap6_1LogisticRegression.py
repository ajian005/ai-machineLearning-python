"""
逻辑回归（Logistic Regression）
逻辑回归是一种经典的线性分类模型，虽然名字中有"回归"，但它实际上用于二分类问题。它通过将线性回归的输出通过逻辑函数（sigmoid）映射到 0 和 1 之间，从而预测事件的概率。
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
# 加载数据
data = load_iris()
X, y = data.data, data.target

# 假设 X 是特征矩阵，y 是标签
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 打印模型精度
print(f"Model accuracy: {model.score(X_test, y_test)}")