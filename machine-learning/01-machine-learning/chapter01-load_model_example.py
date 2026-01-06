# 独立的模型加载示例
import numpy as np
import joblib

# 加载保存的模型
model_filename = 'house_price_model.pkl'
loaded_model = joblib.load(model_filename)

print("模型加载成功！")
print(f"模型类型：{type(loaded_model)}")

# 使用加载的模型进行预测
def predict_house_price(area):
    """预测房屋价格
    参数：
        area: 房屋面积（平方米）
    返回：
        预测价格（万元）
    """
    prediction = loaded_model.predict([[area]])
    return prediction[0]

# 测试几个不同的房屋面积
test_areas = [75, 95, 125, 150]
print("\n房屋价格预测：")
print("-" * 30)
for area in test_areas:
    price = predict_house_price(area)
    print(f"{area} 平方米：{price:.2f} 万元")

# 显示模型的参数
print(f"\n模型参数：")
print(f"斜率（系数）：{loaded_model.coef_[0]:.2f}")
print(f"截距：{loaded_model.intercept_:.2f}")
print(f"价格公式：价格 = {loaded_model.coef_[0]:.2f} × 面积 + {loaded_model.intercept_:.2f}")