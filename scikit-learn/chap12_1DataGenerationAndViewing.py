"""
1. 数据生成与查看

使用一个字典创建了模拟数据，并将其转换为 pandas DataFrame。
数据包含了房屋的面积、房间数、楼层、建造年份、位置（类别变量），以及房价（目标变量）。
通过 df.head() 和 df.describe() 来检查数据的基本结构和统计信息。

1、数据生成与查看
我们首先构造一个模拟的 DataFrame，其中包含一些常见的房价预测特征，如房屋面积、房间数、楼层、建造年份和地理位置（类别型变量）。
"""

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
print("统计信息预览：")
print(df.describe)

