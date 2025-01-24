"""
使用 cross_val_score 执行 K-fold 交叉验证
cross_val_score 函数用于执行 K-fold 交叉验证，返回每个折叠的评分结果，帮助我们评估模型的稳定性和性能。

使用 cross_val_score 执行 K-fold 交叉验证:

"""

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# 加载数据
data = load_iris()
X, y = data.data, data.target

# 创建模型
model = RandomForestClassifier()

# 执行 K-fold 交叉验证
scores = cross_val_score(model, X, y, cv=5)  # 5-fold 交叉验证
print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {scores.mean()}")