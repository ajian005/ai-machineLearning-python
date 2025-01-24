"""
鸢尾花数据集（Iris Dataset）是机器学习中最经典的入门数据集之一。

鸢尾花数据集包含了三种鸢尾花（Setosa、Versicolor、Virginica）每种花的 4 个特征：花萼长度、花萼宽度、花瓣长度和花瓣宽度。

接下来我们的任务是基于这些特征来预测鸢尾花的种类。

本章节案例将涵盖数据加载、可视化、特征选择、数据预处理、建立分类模型、模型评估及优化等步骤。

数据加载：

使用 load_iris() 加载鸢尾花数据集，并将数据转换为 DataFrame 格式，以便查看和分析。
数据可视化：

使用 seaborn 的 pairplot 绘制各特征之间的散点图矩阵，并通过 heatmap 绘制特征之间的相关性热力图。
特征选择与数据预处理：

提取特征（X）和标签（y），并对特征数据进行标准化处理，使得每个特征的均值为 0，方差为 1。
建立分类模型：

使用 DecisionTreeClassifier 和 SVC 分别训练决策树分类器和支持向量机分类器，并评估它们在测试集上的准确率。
模型评估：

使用 classification_report 和 confusion_matrix 输出模型的详细评估指标，包括精度、召回率、F1 分数以及混淆矩阵。
网格搜索调优：

使用 GridSearchCV 对决策树模型进行超参数调优，寻找最佳的超参数组合，并输出优化后的模型准确率。
交叉验证：

使用 cross_val_score 进行 5 折交叉验证，评估优化后的决策树模型的稳定性和表现。

涵盖了鸢尾花数据集的加载、数据预处理、特征选择、建立分类模型、模型评估与优化等步骤。我们将使用决策树和 SVM 分类器，并通过网格搜索优化模型超参数。

"""

# 导入必要的库
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. 数据加载
# 加载鸢尾花数据集
data = load_iris()

# 转换为 DataFrame 方便查看
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df['species'] = df['target'].apply(lambda x: data.target_names[x])

# 查看前几行数据
print("数据预览：")
print(df.head())

# 2. 数据可视化
# 绘制特征之间的关系
sns.pairplot(df, hue="species")
plt.show()

# 绘制热力图查看特征之间的相关性
correlation_matrix = df.drop(columns=['target', 'species']).corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# 3. 特征选择与数据预处理
# 提取特征和标签
X = df.drop(columns=['target', 'species'])
y = df['target']

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 建立分类模型
# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 使用决策树分类器
model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train, y_train)

# 预测
y_pred_dt = model_dt.predict(X_test)

# 输出决策树的准确率
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {accuracy_dt:.4f}")

# 使用支持向量机（SVM）分类器
model_svm = SVC(kernel='linear', random_state=42)
model_svm.fit(X_train, y_train)

# 预测
y_pred_svm = model_svm.predict(X_test)

# 输出SVM的准确率
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {accuracy_svm:.4f}")

# 5. 模型评估
# 决策树模型评估
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))

print("\nDecision Tree Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))

# SVM模型评估
print("\nSVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

print("\nSVM Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))

# 6. 网格搜索调优
# 定义决策树的参数网格
param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 初始化 GridSearchCV
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 获取最佳参数和最佳模型
print("\nBest Parameters from GridSearchCV (Decision Tree):")
print(grid_search.best_params_)

# 使用最佳模型进行预测
best_model = grid_search.best_estimator_
y_pred_optimized = best_model.predict(X_test)

# 输出优化后的决策树准确率
accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
print(f"Optimized Decision Tree Accuracy: {accuracy_optimized:.4f}")

# 7. 交叉验证
# 进行 5 折交叉验证
cross_val_scores = cross_val_score(best_model, X_scaled, y, cv=5)
print("\nCross-validation Scores (Optimized Decision Tree):")
print(cross_val_scores)
print(f"Mean CV Accuracy: {cross_val_scores.mean():.4f}")