### Sklearn 基础概念
在使用 Sklearn 进行机器学习时，需要了解一些基础的概念。

Sklearn 提供了一种统一且简洁的 API 来实现各种机器学习算法和流程，能够帮助我们快速实现各种机器学习任务。

接下来我们从以下几个概念展开说明：数据表示、模型类型、预处理方法、评估指标、模型调优等。
***
### 1、数据表示：数据集和特征
数据集是 Sklearn 最基本的概念之一。

机器学习的核心任务是从数据中学习模式，数据的表示方式至关重要。

#### 数据集（Dataset）
在 scikit-learn 中，数据通常通过两个主要的对象来表示： 特征矩阵 和 目标向量。

**特征矩阵（Feature Matrix）**：每一行代表一个数据样本，每一列代表一个特征（即输入变量）。它是一个二维的数组或矩阵，通常使用 NumPy 数组或 pandas DataFrame 来存储。

假设我们有 3 个样本，每个样本有 2 个特征。

实例
```python
import numpy as np
X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
```
**目标向量（Target Vector）**：它表示每个样本的目标（即输出标签），通常是一个一维数组。

例如，在分类任务中，目标是每个样本的类别标签。

对应的目标向量:

`y = np.array([0, 1, 0])  # 0 类别和 1 类别`


#### 特征和标签
**特征（Features）**：是数据集中用于训练模型的输入变量。在上面的例子中，X 是特征矩阵，包含了所有的输入变量。

**标签（Labels）**：是机器学习模型的目标输出。在监督学习中，标签是我们希望模型预测的结果。在上面的例子中，y 是标签或目标向量，包含了每个样本的类别。

#### 数据集分割
在实际应用中，通常需要将数据集分割成训练集和测试集。

scikit-learn 提供了一个方便的函数 train_test_split() 来实现这一点：

实例
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```
- 以上代码调用 train_test_split 函数，并将结果赋值给四个变量：X_train、X_test、y_train 和 y_test。
- X 和 y 是传入 train_test_split 函数的参数，它们分别代表特征数据集和目标变量（标签）。通常 X 是一个二维数组，y 是一个一维数组。
- test_size=0.3 参数指定了测试集的大小应该是原始数据集的 30%。这意味着 70% 的数据将被用作训练集，剩下的 30% 将被用作测试集。
- random_state=42 参数是一个随机数种子，用于确保每次分割数据集时都能得到相同的结果。这在实验和模型验证中非常有用，因为它确保了结果的可重复性。

***
### 2 模型与算法
#### 监督学习和无监督学习
在 scikit-learn 中，机器学习模型大致分为两大类：监督学习 和 无监督学习。

**监督学习Supervised Learning**:在监督学习中，模型在训练时会利用带标签的数据进行学习，这些标签是我们希望模型预测的结果。

常见的监督学习任务包括分类和回归。
  - 分类（Classification）：将数据点分配到预定的类别中。例如，判断邮件是垃圾邮件还是非垃圾邮件。
  - 回归（Regression）：预测连续值输出。例如，预测房价、气温等。

使用决策树进行分类任务：
实例
```python
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```
**无监督学习（Unsupervised Learning）**：无监督学习是指没有标签数据，模型仅通过输入数据本身的特征进行学习。

常见的无监督学习任务包括聚类和降维。

  - 聚类（Clustering）：将数据分组，使得同一组中的数据具有相似性。常见的聚类算法包括 K-Means、DBSCAN 等。
  - 降维（Dimensionality Reduction）：减少数据中的特征数量，通常用于数据压缩或可视化。常见的降维方法有 PCA（主成分分析）和 t-SNE（t-分布随机邻域嵌入）等。

使用 K-Means 聚类：

实例
```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_train)
y_pred = kmeans.predict(X_test)
```

#### 预处理与特征工程
在使用 scikit-learn 进行机器学习之前，通常需要对数据进行预处理，这包括以下几项常见任务：

**1、标准化（Standardization）**：特征的尺度统一，使得每个特征都具有零均值和单位方差。

实例
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
**2、归一化（Normalization）**：将特征的值缩放到一个固定范围（通常是 0 到 1）。

实例
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
```

**3、类别变量编码**：将类别型数据转换为数值型数据（如 one-hot 编码）。
***

### 3、模型评估与验证
在训练机器学习模型之后，必须评估其性能以确保其泛化能力。

scikit-learn 提供了一些评估模型性能的工具。

#### 交叉验证（Cross-validation）
交叉验证是一种常见的模型评估方法，尤其是在数据量有限的情况下。

通过将数据分成多个子集，每次使用一个子集作为验证集，其余作为训练集，重复多次训练和评估模型，最终计算模型的平均性能。

实例
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X, y, cv=5)  # 5-fold cross-validation
print("Cross-validation scores:", scores)
```

#### 常见评估指标
分类任务的评估指标：

 - 准确率（Accuracy）：预测正确的样本占所有样本的比例。
 - 精确率（Precision）：正类预测中，实际正类的比例。
 - 召回率（Recall）：实际正类中，正确预测的比例。
 - F1 分数：精确率和召回率的调和平均数。

实例
```python
from sklearn.metrics import accuracy_score, classification_report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

回归任务的评估指标：

 - 均方误差（MSE）：预测值与真实值的平方差的平均值。
 - 决定系数（R²）：衡量模型对数据变异的解释能力。

实例
```python
from sklearn.metrics import mean_squared_error, r2_score
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))
```

***

### 4、模型选择与调优
#### 网格搜索（Grid Search）
网格搜索是一种常用的超参数调优方法，它通过遍历所有可能的参数组合来寻找最佳的超参数组合。

实例
```python
from sklearn.model_selection import GridSearchCV

param_grid = {'max_depth': [3, 5, 7], 'min_samples_split': [2, 5, 10]}
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)
```

#### 随机搜索（Random Search）
随机搜索是一种通过随机选择超参数的组合来搜索最优超参数的方法，它比网格搜索效率高。

实例
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {'max_depth': [3, 5, 7], 'min_samples_split': randint(2, 10)}
random_search = RandomizedSearchCV(DecisionTreeClassifier(), param_dist, n_iter=10, cv=5)
random_search.fit(X_train, y_train)
print("Best parameters:", random_search.best_params_)
