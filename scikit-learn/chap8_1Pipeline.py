"""
创建一个简单的 Pipeline
假设我们有一个数据集，并且需要对数据进行标准化后训练一个支持向量机（SVM）分类器。我们可以将标准化和模型训练步骤组合成一个管道（Pipeline）。

"""


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# 加载数据
data = load_iris()
X, y = data.data, data.target

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建 Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # 数据标准化
    ('svc', SVC())  # 支持向量机分类器
])

# 训练模型
pipeline.fit(X_train, y_train)

# 预测结果
y_pred = pipeline.predict(X_test)

# 打印模型精度
print(f"Model accuracy: {pipeline.score(X_test, y_test)}")