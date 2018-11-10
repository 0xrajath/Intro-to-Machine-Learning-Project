import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.ensemble import GradientBoostingClassifier

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

param_grid = {
    'max_depth' : [10, 16],
    'n_estimators' : [10, 50, 100],
    'learning_rate' : [0.01, 0.05, 0.1, 0.5]
}

model = GridSearchCV(GradientBoostingClassifier(), param_grid)
model.fit(data_train, label_train)
