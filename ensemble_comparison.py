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

N_MODELS = 6
model_names = ['Decision Tree', 'Bagging', 'Random Forest', 'Adaboost', 'Gradient Boosting', 'XGBoost']
models = [
          # Decision Tree as baseline for ensemble performance
          tree.DecisionTreeClassifier(random_state=1),
          BaggingClassifier(random_state=1, n_estimators=100),
          RandomForestClassifier(random_state=1, n_estimators=100),
          AdaBoostClassifier(random_state=1, base_estimator=tree.DecisionTreeClassifier(max_depth=10), n_estimators=50, learning_rate=1),
          GradientBoostingClassifier(random_state=1, max_depth=10, n_estimators=100, learning_rate=0.1),
          xgb.XGBClassifier(random_state=1, max_depth=10, n_estimators=100, learning_rate=0.1)
]
accuracy_train = []
accuracy_test = []

#for i in range(len(models)):
for i in range(N_MODELS):
    #Building the model using training set
    models[i].fit(data_train, label_train)
    #Finding prediction using the test set
    prediction = models[i].predict(data_test)
    prediction_train = models[i].predict(data_train)
    
    accuracy_test.append(accuracy_score(label_test, prediction))
    accuracy_train.append(accuracy_score(label_train, prediction_train))

    #Prediction metrics
    print(model_names[i])
    print('Testing accuracy: ', accuracy_test[i])
    print('Training accuracy: ', accuracy_train[i])
    print('Classification Report: ')
    print(classification_report(label_test, prediction))
    print('Confusion Matrix: ')
    print(confusion_matrix(label_test, prediction))
    print('\n')
    print('\n')

print('Accuracy Summary:')
for i in range(N_MODELS):
    print('Tesing accuracy for', model_names[i], 'is', accuracy_test[i])
for i in range(N_MODELS):
    print('Traing accuracy for', model_names[i], 'is', accuracy_train[i])
