"""
加载模型
使用 joblib.load() 方法加载保存的模型对象。
"""

import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# 加载数据
data = load_iris()
X, y = data.data, data.target

# 加载数据
data = load_iris()
X, y = data.data, data.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 加载保存的模型
loaded_model = joblib.load('svm_model.joblib')

# 使用加载的模型进行预测
y_pred = loaded_model.predict(X_test)

# 打印预测结果
print("Predictions:", y_pred)