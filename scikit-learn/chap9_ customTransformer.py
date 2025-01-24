"""
自定义转换器示例：自定义标准化转换器：

假设我们希望实现一个自定义的标准化转换器，标准化是将数据的每个特征缩放到均值为 0、方差为 1 的范围。

"""


import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CustomScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        """
        计算每个特征的均值和标准差
        """
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self # 返回对象本身
    
    def transform(self, X):
        """
        标准化数据
        """
        return (X - self.mean_) / self.std_
    
# 测试自定义转换器
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 加载数据
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建自定义转换器对象
scaler = CustomScaler()

# 使用自定义标准化
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaled training data:\n", X_train_scaled)