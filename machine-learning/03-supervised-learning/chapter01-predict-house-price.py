from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# 假设我们有一个简单的房价数据集
data = {
    '面积': [50, 60, 80, 100, 120],
    '房价': [150, 180, 240, 300, 350]
}
df = pd.DataFrame(data)

# 特征和标签
X = df[['面积']]
y = df['房价']

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

print(f"预测的房价: {y_pred}")