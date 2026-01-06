# 导入需要的库
from inspect import getmodule
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import joblib
import os

# 设置图表风格，让图表更好看
sns.set_style("whitegrid")
# -------------------------- 设置中文字体 start --------------------------
plt.rcParams['font.sans-serif'] = [
        # Windows 优先
        'SimHei', 'Microsoft YaHei',
        # macOS 优先
        'PingFang SC', 'Heiti TC',
        # Linux 优先
        'WenQuanYi Micro Hei', 'DejaVu Sans'
]
# 修复负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
# -------------------------- 设置中文字体 end --------------------------
# 1. 准备数据
# 假设我们有房屋面积和对应的价格数据
# 房屋面积（平方米）
house_sizes = np.array([50, 60, 70, 80, 90, 100, 110, 120]).reshape(-1, 1)
# 房屋价格（万元）
house_prices = np.array([150, 180, 210, 240, 270, 300, 330, 360])


# 2. 创建并训练模型
# 创建线性回归模型
model = LinearRegression()
# 用数据训练模型（学习面积和价格之间的关系）
model.fit(house_sizes, house_prices)

get_params =   model.get_params()
print("model get_params:", get_params)

model_score = model.score(house_sizes, house_prices)
print("model score:", model_score)

model_coef = model.coef_
print("model coef:", model_coef)

model_intercept = model.intercept_
print("model intercept:", model_intercept)

# 3. 使用模型进行预测
# 预测 85 平方米的房屋价格
predicted_price = model.predict([[85]])
print(f"85 平方米的房屋预测价格：{predicted_price[0]:.2f} 万元")

# 4. 保存模型到文件
model_filename = './machine-learning/01-machine-learning/chapter01_house_price_model.pkl'
joblib.dump(model, model_filename)
print(f"模型已保存到文件：{model_filename}")

# 5. 加载模型（演示）
# 在实际应用中，你可以在另一个程序中加载这个模型
loaded_model = joblib.load(model_filename)
print("模型已从文件加载")

# 使用加载的模型进行预测
loaded_prediction = loaded_model.predict([[85]])
print(f"使用加载的模型预测 85 平方米房屋价格：{loaded_prediction[0]:.2f} 万元")

# 验证加载的模型与原模型预测结果一致
print(f"原模型与加载模型预测结果一致：{abs(predicted_price[0] - loaded_prediction[0]) < 0.001}")


# 6. 可视化结果
plt.scatter(house_sizes, house_prices, color='blue', label='实际数据')
plt.plot(house_sizes, model.predict(house_sizes), color='red', label='预测线')
plt.scatter([85], predicted_price, color='green', s=100, label='预测点')
plt.xlabel('房屋面积（平方米）')
plt.ylabel('房屋价格（万元）')
plt.title('RUNOOB 机器学习测试 -- 房屋面积与价格关系')
plt.legend()
plt.grid(True)
plt.show()


