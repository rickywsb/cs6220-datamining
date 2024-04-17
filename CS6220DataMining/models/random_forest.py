import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor # 使用回归模型，因为 p_recall 是一个连续值
from sklearn.metrics import mean_squared_error
from math import sqrt


df = pd.read_csv('../data/Duolingo_dataset_outliers_capped.csv')



X = df.drop('p_recall', axis=1)
y = df['p_recall']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("split done")

model = RandomForestRegressor(n_estimators=100, random_state=42)


model.fit(X_train, y_train)
print("model train")

predictions = model.predict(X_test)
print("model predict")

rmse = sqrt(mean_squared_error(y_test, predictions))
print(f"RMSE: {rmse}")
