# Sklearn 自定义模型与功能

自定义模型可以扩展 scikit-learn 的功能，以便满足特定应用场景下的需求。

在 scikit-learn 中，除了使用现成的模型和预处理功能外，用户还可以根据自己的需求创建自定义的模型、转换器和功能。

自定义模型和功能的实现通常涉及继承 scikit-learn 的基类，如 BaseEstimator 和 TransformerMixin，然后实现特定的 fit 和 predict 方法。

本章节会在以下几个方面展开说明：

 - 自定义转换器（Transformer）
 - 自定义估计器（Estimator）
 - 自定义管道步骤
 - 如何通过继承 BaseEstimator 和 TransformerMixin 实现自定义算法
### 1、自定义转换器（Transformer）
转换器是用于对数据进行转换的组件，例如标准化、特征选择等。自定义转换器可以继承 TransformerMixin，并实现 fit 和 transform 方法。

***自定义转换器的步骤***：

 - **fit 方法**：通常用于学习数据的属性（如均值、方差、特征选择标准等）。fit 方法返回转换器本身，以便可以进行链式调用。
 - **transform 方法**：应用已学习的属性，对数据进行转换或处理。

***自定义转换器示例：自定义标准化转换器***：

假设我们希望实现一个自定义的标准化转换器，标准化是将数据的每个特征缩放到均值为 0、方差为 1 的范围。

实例
```python
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CustomScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        """
        计算每个特征的均值和标准差
        """
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self  # 返回对象本身

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
```
输出如下：

Scaled training data:
```
 [[-1.47393679  1.20365799 -1.56253475 -1.31260282]
 [-0.13307079  2.99237573 -1.27600637 -1.04563275]
 [ 1.08589829  0.08570939  0.38585821  0.28921757]
 [-1.23014297  0.75647855 -1.2187007  -1.31260282]
 [-1.7177306   0.30929911 -1.39061772 -1.31260282]
....
```
**说明：**

CustomScaler 类实现了一个标准化过程，类似于 scikit-learn 中的 StandardScaler。

 - fit 方法计算训练数据的均值和标准差，并保存这些值。
 - transform 方法根据 fit 中计算的均值和标准差来转换数据。

***

### 2、自定义估计器（Estimator）
估计器是指模型本身，如回归器、分类器等。

自定义估计器需要继承 BaseEstimator 类，并实现 fit 和 predict 方法。

***自定义估计器的步骤:***

 - fit 方法：用于训练模型，计算需要的参数（如权重、偏置等）。
 - predict 方法：基于训练好的参数对输入数据进行预测。
#### 自定义估计器示例：简单的分类器
假设我们要实现一个非常简单的分类器：将每个特征的均值作为阈值，超过均值的预测为类别 1，否则预测为类别 0。

实例
```python
from sklearn.base import BaseEstimator
import numpy as np

class SimpleClassifier(BaseEstimator):
    def fit(self, X, y):
        """
        训练模型：计算每个特征的均值
        """
        self.mean_ = np.mean(X, axis=0)
        return self  # 返回对象本身

    def predict(self, X):
        """
        基于均值进行分类：如果特征值大于均值，则预测为 1，否则为 0
        """
        return (X > self.mean_).astype(int)

# 测试自定义分类器
X_train = np.array([[1.5, 2.5], [2.0, 3.0], [3.5, 4.5], [4.0, 5.0]])
y_train = np.array([0, 0, 1, 1])

# 创建自定义分类器对象
classifier = SimpleClassifier()

# 训练模型
classifier.fit(X_train, y_train)

# 进行预测
X_test = np.array([[2.5, 3.5], [1.0, 2.0]])
y_pred = classifier.predict(X_test)

print("Predictions:", y_pred)
```
输出如下：
```
Predictions: [[0 0]
 [0 0]]
 ```
***说明：***

 - fit 方法计算训练数据的均值并将其存储在 self.mean_ 中。
 - predict 方法通过比较测试数据与均值的大小，做出分类预测。
这种自定义的分类器在实际中并不常用，但它展示了如何使用 BaseEstimator 创建一个简单的模型。

***
### 3、自定义管道步骤
scikit-learn 允许将自定义转换器和估计器作为管道的步骤。这使得可以将数据预处理和模型训练过程整合成一个工作流。自定义模型和功能可以像内建的转换器和估计器一样，与管道一起工作。

***使用自定义转换器和估计器在管道中***
实例
```python
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class CustomScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        """
        计算每个特征的均值和标准差
        """
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self  # 返回对象本身

    def transform(self, X):
        """
        标准化数据
        """
        return (X - self.mean_) / self.std_
   

class SimpleClassifier(BaseEstimator):
    def fit(self, X, y):
        """
        训练模型：计算每个特征的均值
        """
        self.mean_ = np.mean(X, axis=0)
        return self  # 返回对象本身

    def predict(self, X):
        """
        基于均值进行分类：如果特征值大于均值，则预测为 1，否则为 0
        """
        return (X > self.mean_).astype(int)

# 测试自定义分类器
X_train = np.array([[1.5, 2.5], [2.0, 3.0], [3.5, 4.5], [4.0, 5.0]])
y_train = np.array([0, 0, 1, 1])

# 创建管道，包含自定义的标准化和分类器
pipeline = Pipeline([
    ('scaler', CustomScaler()),  # 自定义标准化
    ('classifier', SimpleClassifier())  # 自定义分类器
])

# 训练管道
pipeline.fit(X_train, y_train)

X_test = np.array([[2.5, 3.5], [1.0, 2.0]])
# 预测
y_pred = pipeline.predict(X_test)
print("Predictions:", y_pred)
```
输出如下：
```
Predictions: [[0 0]
 [0 0]]
 ```
说明：

我们将 CustomScaler 和 SimpleClassifier 作为管道的步骤，利用管道自动执行数据预处理和模型训练。

这样，管道的整个工作流可以通过一个 fit 和 predict 方法完成，保持了高效性和简洁性。

***

### 如何通过继承 BaseEstimator 和 TransformerMixin 实现自定义算法
 - BaseEstimator：是所有 scikit-learn 估计器的基类。它提供了 get_params() 和 set_params() 方法，这使得自定义估计器能够与 GridSearchCV 等工具配合工作。
 - TransformerMixin：是所有 scikit-learn 转换器的基类，它提供了 fit_transform() 方法，使得转换器能够与 Pipeline 一起使用。
通过继承这两个基类，我们可以非常方便地创建自己的自定义模型和功能，并且可以利用 scikit-learn 的工具（如 GridSearchCV 和 Pipeline）进行调优和评估。

完整自定义转换器和估计器示例:

实例
```python
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class CustomEstimator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # 模拟一个简单的"模型"：计算每个特征的均值
        self.mean_ = np.mean(X, axis=0)
        return self

    def transform(self, X):
        # 基于均值将数据标准化
        return X - self.mean_

    def predict(self, X):
        # 简单的预测方法：如果特征值大于均值，则预测为 1，否则为 0
        return (X > self.mean_).astype(int)

# 使用自定义估计器和转换器
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 加载数据
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建管道
pipeline = Pipeline([
    ('custom', CustomEstimator())
])

# 训练模型
pipeline.fit(X_train, y_train)

# 预测
y_pred = pipeline.predict(X_test)
print("Predictions:", y_pred)
```
以上例子中，我们创建了一个自定义的估计器 CustomEstimator，它不仅执行标准化（transform），还执行预测（predict）。然后，我们将其作为管道的一部分进行训练和预测。

输出如下所示：
```
Predictions: [[1 0 1 1]
 [0 1 0 0]
 [1 0 1 1]
 [1 0 1 1]
 [1 0 1 1]
 [0 1 0 0]
 [0 0 0 1]
 [1 1 1 1]
 [1 0 1 1]
 [0 0 1 1]
...
]
```