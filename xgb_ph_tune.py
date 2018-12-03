import pandas as pd
import numpy as np

import xgboost as xgb
from sklearn.model_selection import GridSearchCV, cross_val_score

data_loc = 'data/uci_data/poker_hand/poker-hand-training-true.data.txt'
col_names = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'hand']
df = pd.read_csv(data_loc, names=col_names)
df_x = df.drop(col_names[-1], axis=1)
df_y = df[col_names[-1]]

'''
param_test1 = {'n_estimators': range(20, 101, 10)}
gsearch1 = GridSearchCV(
                        estimator=xgb.XGBClassifier(random_state=1, learning_rate=2, max_depth=5, min_child_weight=1, gamma=0),
                        param_grid=param_test1, scoring='accuracy', n_jobs=-1, cv=5)
gsearch1.fit(df_x, df_y)

print('results:')
for mean, std, params in zip(gsearch1.cv_results_['mean_test_score'], gsearch1.cv_results_['std_test_score'], gsearch1.cv_results_['params']):
    print("mean: %0.5f, std: %0.5f, params: %r" % (mean, std, params))
print('\n')

print('best parameter:')
print(gsearch1.best_params_)
print('\n')

print('best score:')
print(gsearch1.best_score_)
'''

'''
param_test2 = {'max_depth': range(3, 10, 2), 'min_child_weight': range(1, 6, 2)}
gsearch2 = GridSearchCV(
                        estimator=xgb.XGBClassifier(random_state=1, learning_rate=2, n_estimators=80, gamma=0),
                        param_grid=param_test2, scoring='accuracy', n_jobs=-1, cv=5)
gsearch2.fit(df_x, df_y)

print('results:')
for mean, std, params in zip(gsearch2.cv_results_['mean_test_score'], gsearch2.cv_results_['std_test_score'], gsearch2.cv_results_['params']):
    print("mean: %0.5f, std: %0.5f, params: %r" % (mean, std, params))
print('\n')

print('best parameter:')
print(gsearch2.best_params_)
print('\n')

print('best score:')
print(gsearch2.best_score_)
'''

'''
param_test3 = {'gamma': [i/10.0 for i in range(0, 5)]}
gsearch3 = GridSearchCV(
                        estimator=xgb.XGBClassifier(random_state=1, learning_rate=2, n_estimators=80, max_depth=5, min_child_weight=3),
                        param_grid=param_test3, scoring='accuracy', n_jobs=-1, cv=5)
gsearch3.fit(df_x, df_y)

print('results:')
for mean, std, params in zip(gsearch3.cv_results_['mean_test_score'], gsearch3.cv_results_['std_test_score'], gsearch3.cv_results_['params']):
    print("mean: %0.5f, std: %0.5f, params: %r" % (mean, std, params))
print('\n')

print('best parameter:')
print(gsearch3.best_params_)
print('\n')

print('best score:')
print(gsearch3.best_score_)
'''

'''
param_test4 = {'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]}
gsearch4 = GridSearchCV(
                        estimator=xgb.XGBClassifier(random_state=1, learning_rate=2, n_estimators=80, max_depth=5, min_child_weight=3, gamma=0),
                        param_grid=param_test4, scoring='accuracy', n_jobs=-1, cv=5)
gsearch4.fit(df_x, df_y)

print('results:')
for mean, std, params in zip(gsearch4.cv_results_['mean_test_score'], gsearch4.cv_results_['std_test_score'], gsearch4.cv_results_['params']):
    print("mean: %0.5f, std: %0.5f, params: %r" % (mean, std, params))
print('\n')

print('best parameter:')
print(gsearch4.best_params_)
print('\n')

print('best score:')
print(gsearch4.best_score_)
'''

param_test5 = {'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05]}
gsearch5 = GridSearchCV(
                        estimator=xgb.XGBClassifier(random_state=1, learning_rate=2, n_estimators=80, max_depth=5, min_child_weight=3, gamma=0),
                        param_grid=param_test5, scoring='accuracy', n_jobs=-1, cv=5)
gsearch5.fit(df_x, df_y)

print('results:')
for mean, std, params in zip(gsearch5.cv_results_['mean_test_score'], gsearch5.cv_results_['std_test_score'], gsearch5.cv_results_['params']):
    print("mean: %0.5f, std: %0.5f, params: %r" % (mean, std, params))
print('\n')

print('best parameter:')
print(gsearch5.best_params_)
print('\n')

print('best score:')
print(gsearch5.best_score_)
