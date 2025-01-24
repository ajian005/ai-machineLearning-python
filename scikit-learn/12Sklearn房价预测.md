### Sklearn 房价预测
接下来我们使用 sklearn 进行北京房价预测。

我们会从数据加载、探索性分析、特征工程到模型训练与优化逐步展开，演示如何使用 sklearn 的工具和库完成预测任务。

内容概要：

#### 1. 数据生成与查看

 - 使用一个字典创建了模拟数据，并将其转换为 pandas DataFrame。
 - 数据包含了房屋的面积、房间数、楼层、建造年份、位置（类别变量），以及房价（目标变量）。
 - 通过 df.head() 和 df.describe() 来检查数据的基本结构和统计信息。
#### 2. 数据预处理

 - 特征选择：从原始数据中选择了与房价相关的特征（面积、房间数、楼层、建造年份、位置）。
 - 数据拆分：使用 train_test_split 将数据集分为 80% 的训练集和 20% 的测试集。
 - 数值特征标准化：使用 StandardScaler 对数值特征进行标准化。
 - 类别特征编码：使用 OneHotEncoder 对类别特征（location）进行 One-Hot 编码。
 - ColumnTransformer 将数值和类别特征的处理整合成一个步骤。
#### 3. 模型训练

 - 使用 Pipeline 将数据预处理与模型训练步骤结合，确保整个过程的流水线化。
 - 使用线性回归模型 (LinearRegression) 进行训练。
#### 4. 模型评估

 - 通过 mean_squared_error 计算均方误差（MSE），以及通过 r2_score 计算决定系数（R²）。
 - 输出评估结果，查看模型的预测准确性。
#### 5. 模型优化

 - 使用 GridSearchCV 对线性回归的超参数进行调优，主要调整的是 fit_intercept（是否拟合截距）。
 - 通过网格搜索找到最佳超参数，并使用最佳模型对测试集进行预测。
 - 重新计算优化后的模型评估指标（MSE 和 R²）。
 ***

 ### 1、数据生成与查看
我们首先构造一个模拟的 DataFrame，其中包含一些常见的房价预测特征，如房屋面积、房间数、楼层、建造年份和地理位置（类别型变量）。

实例
```python
import pandas as pd
import numpy as np

# 模拟数据：房屋面积 (平方米)、房间数、楼层、建造年份、位置（类别变量）
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
```
输出：

数据预览：
```
   area  rooms  floor  year_built location    price
0    70      2      5        2005   Chaoyang  5000000
1    85      3      2        2010   Haidian  6000000
2   100      3      8        2012   Chaoyang  6500000
3   120      4     10        2015  Dongcheng  7000000
4    60      2      3        2000   Fengtai  4500000
```
***
 ### 2、数据预处理
数据预处理通常包括特征选择、特征转换、缺失值处理、数据标准化等。我们将对数值特征进行标准化，对类别特征进行独热编码。

实例
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 特征选择
X = df[['area', 'rooms', 'floor', 'year_built', 'location']]  # 特征
y = df['price']  # 目标变量

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建预处理步骤
numeric_features = ['area', 'rooms', 'floor', 'year_built']
categorical_features = ['location']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())  # 数值特征标准化
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # 处理测试集中的新类别
])

# 组合成 ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# 查看数据预处理后的结构
X_train_transformed = preprocessor.fit_transform(X_train)
print("预处理后的训练数据：")
print(X_train_transformed)
```
输出结果为：

xxx@Mac-mini runoob-test % python3 test.py
预处理后的训练数据：
```
[[ 0.89826776  1.0440738   1.14636101  0.96800387  0.          0.
   0.          1.        ]
 [-0.95622052 -1.23390539 -0.98640366 -1.04544418  1.          0.
   0.          0.        ]
 [-0.72440948 -0.474579   -0.55985073 -0.58080232  0.          0.
   1.          0.        ]
 [-0.26078741 -0.474579   -0.34657426  0.03872015  1.          0.
   0.          0.        ]
 [-0.02897638  0.2847474   0.29325514  0.65824263  0.          0.
   0.          1.        ]
 [-1.18803155 -1.23390539 -1.4129566  -1.81984727  0.          0.
   1.          0.        ]
 [ 0.20283466  0.2847474   0.07997868  0.50336201  0.          1.
   0.          0.        ]
 [ 2.05732294  1.80340019  1.78619041  1.2777651   1.          0.
   0.          0.        ]]
```
***
### 3、建立模型
接下来，我们使用线性回归模型来预测房价，我们使用 Pipeline 来集成预处理和模型训练的步骤。

实例
```python
from sklearn.linear_model import LinearRegression

# 构建一个包含预处理和回归模型的 Pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),  # 数据预处理步骤
    ('regressor', LinearRegression())  # 回归模型
])

# 训练模型
model_pipeline.fit(X_train, y_train)

# 进行预测
y_pred = model_pipeline.predict(X_test)

# 输出预测结果
print("\n预测结果：")
print(y_pred)
```
输出结果为：

预测结果：
```
[6375000.00000001 4874999.99999998]
```
***
### 4、模型评估
在模型评估中，我们通常使用 均方误差（MSE） 和 决定系数（R²） 来评估回归模型的表现。

实例
```python
from sklearn.metrics import mean_squared_error, r2_score

# 计算均方误差（MSE）和决定系数（R²）
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 输出评估结果
print("\n模型评估：")
print(f"均方误差 (MSE): {mse:.2f}")
print(f"决定系数 (R²): {r2:.2f}")
```
输出结果为：

预测结果：
```
[6375000.00000001 4874999.99999998]
```
xxx@Mac-mini runoob-test % python3 test.py

模型评估：
均方误差 (MSE): 648125000000.03
决定系数 (R²): -63.81

***
### 5、模型优化（网格搜索）
为了提高模型的性能，我们可以使用 GridSearchCV 对模型的超参数进行调优。在这个例子中，我们不调整线性回归的参数，因为它没有太多超参数可调。对于更复杂的模型（例如随机森林、XGBoost等），我们可以通过网格搜索找到最佳参数。

实例
```python
from sklearn.model_selection import GridSearchCV

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
```
输出结果为：
```
Fitting 5 folds for each of 2 candidates, totalling 10 fits

最佳参数：
{'regressor__fit_intercept': True}

优化后的模型评估：
均方误差 (MSE): 648125000000.03
决定系数 (R²): -63.81
```


### 完整代码
北京房价预测完整代码:

实例
```python
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
```
运行此代码后输出：

数据预览：
```
   area  rooms  floor  year_built   location    price
0    70      2      5        2005   Chaoyang  5000000
1    85      3      2        2010    Haidian  6000000
2   100      3      8        2012   Chaoyang  6500000
3   120      4     10        2015  Dongcheng  7000000
4    60      2      3        2000    Fengtai  4500000

预测结果：
[6375000.00000001 4874999.99999998]

模型评估：
均方误差 (MSE): 648125000000.03
决定系数 (R²): -63.81
Fitting 5 folds for each of 2 candidates, totalling 10 fits

最佳参数：
{'regressor__fit_intercept': True}

优化后的模型评估：
均方误差 (MSE): 648125000000.03
决定系数 (R²): -63.81
```



