import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv(r"D:\AutoEncoder\selfdataset\Dataset.csv")

X = data.drop(labels=['Label',
                      'Flow ID',
                      'Src IP',
                      'Dst IP'], axis=1)
y = data['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=5, loss_function='MultiClass')

model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=10, verbose=False)

feature_importance = pd.DataFrame(list(zip(X.columns, model.get_feature_importance())), columns=['Feature', 'Value'])

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(feature_importance)