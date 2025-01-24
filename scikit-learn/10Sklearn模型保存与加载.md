## Sklearn 模型保存与加载
在机器学习中，模型的训练过程通常是耗时的，为了避免每次重新训练模型，我们可以将训练好的模型保存下来，便于以后进行加载和预测。

scikit-learn 提供了两种常用的方式来保存和加载模型：joblib 和 pickle。
***

### 1、使用 joblib 保存与加载模型
joblib 是一个高效的 Python 序列化工具，特别适合用于保存包含大量数值数组（如 numpy 数组、scikit-learn 模型等）的对象。相较于 pickle，joblib 在处理大规模数据时更高效。

joblib 是 Python 的一个外部库，可以通过以下命令安装：
```
pip install joblib
```
#### 保存模型
joblib 提供了一个简单的 API 来保存和加载对象。

我们可以使用 joblib.dump() 方法将模型保存到文件中。

实例
```python
import joblib
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

# 保存模型到文件
joblib.dump(model, 'svm_model.joblib')
```
### 加载模型
使用 joblib.load() 方法加载保存的模型对象。

实例
```python
# 加载保存的模型
loaded_model = joblib.load('svm_model.joblib')

# 使用加载的模型进行预测
y_pred = loaded_model.predict(X_test)

# 打印预测结果
print("Predictions:", y_pred)
```
通过上述步骤，我们成功地将训练好的模型保存到文件中，并在之后的任何时间加载该模型并进行预测。

***
### 2、使用 pickle 保存与加载模型
pickle 是 Python 内置的模块，允许将 Python 对象序列化和反序列化。

虽然 joblib 更适用于处理大量数据，但 pickle 也是常用的保存和加载模型的工具，适用于一般情况。

#### 保存模型
与 joblib 类似，pickle 也有简单的 API 来保存和加载对象。

保存模型的代码如下：

实例
```python
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
```

#### 加载模型
使用 pickle.load() 加载模型：

实例
```
# 使用 pickle 加载保存的模型
with open('svm_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# 使用加载的模型进行预测
y_pred = loaded_model.predict(X_test)

# 打印预测结果
print("Predictions:", y_pred)
```

***

### 3、joblib vs pickle
joblib 和 pickle 是保存和加载模型的两种常用方法。

joblib 更适合保存大型数据对象，而 pickle 是 Python 的标准序列化工具，适用于一般情况。

 - joblib：通常适用于保存包含大量数值数据（如 numpy 数组）的对象。joblib 在处理大规模数据时比 pickle 更高效。
 - pickle：适用于保存较小的对象或常规的 Python 对象。它是 Python 的内置库，使用时无需额外安装。
如果模型中包含大量数值数组或矩阵（如支持向量机、随机森林等），推荐使用 joblib，它比 pickle 更高效。对于较小的模型或不包含大量数值数据的模型，pickle 足够使用。

***

### 4、保存和加载管道（Pipeline）
在实际应用中，模型不仅仅是单一的模型，有时会结合多个处理步骤（如数据预处理、特征选择、模型训练等），这些处理步骤可以使用 scikit-learn 的 Pipeline 来完成。Pipeline 也可以通过 joblib 或 pickle 保存和加载。

#### 保存管道:

实例
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

# 创建一个管道
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='linear'))
])

# 训练管道
pipeline.fit(X_train, y_train)

# 保存管道到文件
joblib.dump(pipeline, 'pipeline_model.joblib')
```
#### 加载管道:

实例
```python
# 加载管道
loaded_pipeline = joblib.load('pipeline_model.joblib')

# 使用加载的管道进行预测
y_pred = loaded_pipeline.predict(X_test)

# 打印预测结果
print("Predictions:", y_pred)
```
管道保存和加载的过程与单一模型相同，只需确保保存和加载整个管道对象即可。

### 5、模型版本管理
在机器学习的实际应用中，模型的更新和版本管理至关重要。每次训练模型并保存时，最好为模型文件命名加上时间戳或版本号，以便区分不同版本的模型。例如：

实例
```python
import time

# 创建时间戳
timestamp = time.strftime("%Y%m%d-%H%M%S")

# 保存带时间戳的模型
joblib.dump(model, f'svm_model_{timestamp}.joblib')
```
这样，我们可以根据时间戳来管理不同版本的模型，便于模型的回溯和更新。


### 6、使用模型进行持久化
一旦模型训练完成并保存，我们可以在后续的实际应用中加载该模型来进行预测，而无需重新训练。

例如，我们可以将保存的模型与 Web 服务、批处理作业或其他应用程序集成，使得模型可以反复使用，而无需重新训练。

#### Web 服务中使用加载的模型
例如，假设我们正在使用 Flask 创建一个简单的 Web 服务，通过 API 接口提供模型预测服务。在这种情况下，我们可以加载保存的模型进行实时预测。

实例
```python
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# 加载模型
model = joblib.load('svm_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # 获取输入数据
    features = np.array(data['features']).reshape(1, -1)  # 转换成适合预测的格式
    prediction = model.predict(features)  # 使用加载的模型进行预测
    return jsonify({'prediction': prediction.tolist()})  # 返回预测结果

if __name__ == '__main__':
    app.run(debug=True)
```

