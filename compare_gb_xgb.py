import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from timeit import default_timer as timer

from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb

df = pd.read_csv("data/poker_hand_train.csv")
#df = pd.read_csv("data/c4_game_database.csv")
#df = df.fillna(0)

label_name = 'hand'
#label_name = 'winner'
df_x = df.drop(label_name, axis=1)
df_y = df[label_name]
train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=0.3, random_state=0)

N_MODELS = 8
model_names = ['XGBoost(3)', 'XGBoost(10)', 'XGBoost(20)', 'XGBoost(30)', 'Gradient Boosting(3)', 'Gradient Boosting(10)', 'Gradient Boosting(20)', 'Gradient Boosting(30)']
models = [
          xgb.XGBClassifier(random_state=1, max_depth=3, n_estimators=100, learning_rate=0.01),
          xgb.XGBClassifier(random_state=1, max_depth=10, n_estimators=100, learning_rate=0.01),
          xgb.XGBClassifier(random_state=1, max_depth=20, n_estimators=100, learning_rate=0.01),
          xgb.XGBClassifier(random_state=1, max_depth=30, n_estimators=100, learning_rate=0.01),
          GradientBoostingClassifier(random_state=1, max_depth=3, n_estimators=100, learning_rate=0.01),
          GradientBoostingClassifier(random_state=1, max_depth=10, n_estimators=100, learning_rate=0.01),
          GradientBoostingClassifier(random_state=1, max_depth=20, n_estimators=100, learning_rate=0.01),
          GradientBoostingClassifier(random_state=1, max_depth=30, n_estimators=100, learning_rate=0.01)
]
accuracy_train = []
accuracy_test = []
train_time = []

for i in range(N_MODELS):
    start_time = timer()
    models[i].fit(train_x, train_y)
    time_used = timer() - start_time
    prediction_test = models[i].predict(test_x)
    prediction_train = models[i].predict(train_x)
    
    train_time.append(time_used)
    accuracy_test.append(accuracy_score(test_y, prediction_test))
    accuracy_train.append(accuracy_score(train_y, prediction_train))

    print(model_names[i])
    print('Training time:', train_time[i])
    print('Training accuracy:', accuracy_train[i])
    print('Testing accuracy:', accuracy_test[i])
    print('Classification report:')
    print(classification_report(test_y, prediction_test))
    print('Confusion matrix:')
    print(confusion_matrix(test_y, prediction_test))
    print('\n\n')

for i in range(N_MODELS):
    print('Training time for', model_names[i], ':', train_time[i])
print('\n')

for i in range(N_MODELS):
    print('Training accuracy for', model_names[i], ':', accuracy_train[i])
print('\n')

for i in range(N_MODELS):
    print('Testing accuracy for', model_names[i], ':', accuracy_test[i])
print('\n')
