import pandas as pd
import xgboost as xgb

# 读取CSV文件
data = pd.read_csv(r"C:\Users\xuhai\Desktop\train24.csv")

X = data.drop(labels=['Label'], axis=1)
y = data['Label']

# 创建XGBoost分类器对象
model = xgb.XGBClassifier()

# 训练模型
model.fit(X, y)

# 获取特征重要性评估结果
feature_importance = model.feature_importances_

Feature_name = data.columns.tolist()
Feature_name.remove(Feature_name[-1])

Feature_dict = dict(zip(Feature_name, feature_importance))
Feature_dict = dict(sorted(Feature_dict.items(), key=lambda item: item[1]))
print(Feature_dict)
