"""
使用 sklearn 进行北京房价预测。

我们会从数据加载、探索性分析、特征工程到模型训练与优化逐步展开，演示如何使用 sklearn 的工具和库完成预测任务。

内容概要：

1. 数据生成与查看

使用一个字典创建了模拟数据，并将其转换为 pandas DataFrame。
数据包含了房屋的面积、房间数、楼层、建造年份、位置（类别变量），以及房价（目标变量）。
通过 df.head() 和 df.describe() 来检查数据的基本结构和统计信息。
2. 数据预处理

特征选择：从原始数据中选择了与房价相关的特征（面积、房间数、楼层、建造年份、位置）。
数据拆分：使用 train_test_split 将数据集分为 80% 的训练集和 20% 的测试集。
数值特征标准化：使用 StandardScaler 对数值特征进行标准化。
类别特征编码：使用 OneHotEncoder 对类别特征（location）进行 One-Hot 编码。
ColumnTransformer 将数值和类别特征的处理整合成一个步骤。
3. 模型训练

使用 Pipeline 将数据预处理与模型训练步骤结合，确保整个过程的流水线化。
使用线性回归模型 (LinearRegression) 进行训练。
4. 模型评估

通过 mean_squared_error 计算均方误差（MSE），以及通过 r2_score 计算决定系数（R²）。
输出评估结果，查看模型的预测准确性。
5. 模型优化

使用 GridSearchCV 对线性回归的超参数进行调优，主要调整的是 fit_intercept（是否拟合截距）。
通过网格搜索找到最佳超参数，并使用最佳模型对测试集进行预测。
重新计算优化后的模型评估指标（MSE 和 R²）。

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 模拟数据：包含房屋的面积、房间数、楼层、建造年份、位置（类别变量），以及房价（目标变量）
data = {
    'area': [70, 85, 100, 120, 60, 150, 200, 80, 95, 110],
    'rooms': [2, 3, 3, 4, 2, 5, 6, 3, 3, 4],
    'floor': [5, 2, 8, 10, 3, 15, 18, 7, 9, 11],
    'year_built': [2005, 2010, 2012, 2015, 2000, 2018, 2020, 2008, 2011, 2016],
    'location': ['Chaoyang', 'Haidian', 'Chaoyang', 'Dongcheng', 'Fengtai', 'Haidian', 'Chaoyang', 'Fengtai', 'Dongcheng', 'Haidian'],
    'price': [5000000, 6000000, 6500000, 7000000, 4500000, 10000000, 12000000, 5500000, 6200000, 7500000]  # 房价（目标变量）
}

# 创建 DataFrame
df = pd.DataFrame(data)

# 查看数据
print("数据预览：")
print(df.head())

# 特征选择
X = df[['area', 'rooms', 'floor', 'year_built', 'location']]  # 特征数据
y = df['price']  # 目标变量

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据预处理：数值特征标准化、类别特征 One-Hot 编码
numeric_features = ['area', 'rooms', 'floor', 'year_built']
categorical_features = ['location']

# 数值特征预处理：标准化
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# 类别特征预处理：One-Hot 编码，设置 handle_unknown='ignore'
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # 处理测试集中的新类别
])

# 合并数值和类别特征的处理步骤
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# 3. 建立模型
# 使用线性回归模型，结合数据预处理步骤
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# 训练模型
model_pipeline.fit(X_train, y_train)

# 进行预测
y_pred = model_pipeline.predict(X_test)

# 输出预测结果
print("\n预测结果：")
print(y_pred)

# 4. 模型评估：计算均方误差（MSE）和 R² 决定系数
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n模型评估：")
print(f"均方误差 (MSE): {mse:.2f}")
print(f"决定系数 (R²): {r2:.2f}")

# 5. 模型优化：使用网格搜索调整超参数
# 对线性回归的超参数进行调优（仅调整 'fit_intercept'）
param_grid = {
    'regressor__fit_intercept': [True, False],  # 是否拟合截距
}

grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_train, y_train)

# 输出最佳参数和结果
print("\n最佳参数：")
print(grid_search.best_params_)

# 使用最佳模型进行预测
best_model = grid_search.best_estimator_
y_pred_optimized = best_model.predict(X_test)

# 输出优化后的评估结果
mse_opt = mean_squared_error(y_test, y_pred_optimized)
r2_opt = r2_score(y_test, y_pred_optimized)

print("\n优化后的模型评估：")
print(f"均方误差 (MSE): {mse_opt:.2f}")
print(f"决定系数 (R²): {r2_opt:.2f}")