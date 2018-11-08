import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import BaggingClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb

df = pd.read_csv("data/train.csv")

train, test = train_test_split(df, test_size=0.3, random_state=0)
x_train=train.drop('hand', axis=1)
y_train=train['hand']

x_test=test.drop('hand', axis=1)
y_test=test['hand']

x_train=pd.get_dummies(x_train)
x_test=pd.get_dummies(x_test)

model_names = ['bagging', 'randomforest', 'adaboost', 'gradientboosting', 'xgboost']
models = [
          BaggingClassifier(tree.DecisionTreeClassifier(random_state=1)),
          RandomForestClassifier(random_state=1, max_features=5),
          AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(random_state=1), n_estimators=100, learning_rate=0.01),
          GradientBoostingClassifier(random_state=1, n_estimators=100, learning_rate=0.01, max_depth=6),
          xgb.XGBClassifier(random_state=1, n_estimators=100, learning_rate=0.01, max_depth=6)
]
scores = []

for i in range(len(models)):
    models[i].fit(x_train, y_train)
    #scores.append(models[i].score(x_test, y_test))
    prediction = models[i].predict(x_test)
    scores.append(accuracy_score(y_test, prediction))

for i in range(len(models)):
    print('Score for', model_names[i], 'is', scores[i])
