import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor # 使用回归模型，因为 p_recall 是一个连续值
from sklearn.metrics import mean_squared_error
from math import sqrt

# 数据加载
df = pd.read_csv('../data/Duolingo_dataset_outliers_capped.csv')

# 预处理数据，例如：编码分类变量，处理时间戳等（根据需要添加）
# 示例中暂时省略这些步骤

# 定义特征变量和目标变量
X = df.drop('p_recall', axis=1)
y = df['p_recall']

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("split done")
# 构建随机森林模型
model = RandomForestRegressor(n_estimators=100, random_state=42)

# 模型训练
model.fit(X_train, y_train)
print("model train")
# 预测
predictions = model.predict(X_test)
print("model predict")
# 模型评估
rmse = sqrt(mean_squared_error(y_test, predictions))
print(f"RMSE: {rmse}")
