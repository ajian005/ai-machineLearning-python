"""
保存模型
joblib 提供了一个简单的 API 来保存和加载对象。
"""

import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# 加载数据
data = load_iris()
X, y = data.data, data.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并训练模型
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# 保存模型到文件
joblib.dump(model, 'svm_model.joblib')