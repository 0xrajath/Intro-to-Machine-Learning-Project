import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb

# Importing Training dataset
df = pd.read_csv("data/poker_hand_train.csv")

# Splitting existing training set as Training set: 30% , Test Set: 70%
train, test = train_test_split(df, test_size=0.3, random_state=0)

# Creating Data Columns and Label Columns
data_train=train.drop('hand', axis=1)
label_train=train['hand']

data_test=test.drop('hand', axis=1)
label_test=test['hand']

data_train=pd.get_dummies(data_train)
data_test=pd.get_dummies(data_test)

model_names = ['Bagging', 'Random Forest', 'Adaboost', 'Gradient Boosting', 'XGBoost']
models = [
          BaggingClassifier(tree.DecisionTreeClassifier(random_state=1)),
          RandomForestClassifier(random_state=1, max_features=5),
          AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(random_state=1), n_estimators=100, learning_rate=0.01),
          GradientBoostingClassifier(random_state=1, n_estimators=100, learning_rate=0.01, max_depth=3),
          xgb.XGBClassifier(random_state=1, n_estimators=100, learning_rate=0.01, max_depth=3)
]
scores = []

for i in range(len(models)):
    #Building the model using training set
    models[i].fit(data_train, label_train)
    #Finding prediction using the test set
    prediction = models[i].predict(data_test)
    
    scores.append(accuracy_score(label_test, prediction))

    #Prediction metrics
    print(model_names[i])
    print('Score: ', accuracy_score(label_test, prediction))
    print('Classification Report: ')
    print(classification_report(label_test, prediction))
    print('Confusion Matrix: ')
    print(confusion_matrix(label_test, prediction))
    print('\n')
    print('\n')
    scores.append(accuracy_score(label_test, prediction))

print('Score Summary:')
for i in range(len(models)):
    print('Score for', model_names[i], 'is', scores[i])

