"""
2、使用 pickle 保存与加载模型
pickle 是 Python 内置的模块，允许将 Python 对象序列化和反序列化。

虽然 joblib 更适用于处理大量数据，但 pickle 也是常用的保存和加载模型的工具，适用于一般情况。

保存模型
与 joblib 类似，pickle 也有简单的 API 来保存和加载对象。

保存模型的代码如下：
"""


import pickle
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

# 使用 pickle 保存模型
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(model, f)