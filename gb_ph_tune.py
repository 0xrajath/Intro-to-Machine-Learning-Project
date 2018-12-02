import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

data_loc = 'data/uci_data/poker_hand/poker-hand-training-true.data.txt'
col_names = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'hand']
df = pd.read_csv(data_loc, names=col_names)
df_x = df.drop(col_names[-1], axis=1)
df_y = df[col_names[-1]]

'''
    There are no optimum values for learning rate as low values always work better, given that we
    train on sufficient number of trees. High learning rates can lead to overfitting. But as we reduce
    the learning rate and increase trees, the computation becomes unreasonably expensive for tuning other
    parameters.
    
    Therefore, we first choose a relatively high learning rate, and determine the optimum number
    of trees for this learning rate. Then after we fine-tune other parameters, we lower the learning rate
    and increase the trees proportionally to get more robust models.
    '''


param_test1 = {'n_estimators': range(20, 101, 10)}
gsearch1 = GridSearchCV(
                        estimator=GradientBoostingClassifier(random_state=1, learning_rate=0.5, min_samples_split=2, min_samples_leaf=1, max_depth=8, max_features='sqrt', subsample=0.8),
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
    We got 50 as the optimal number of trees for 0.5 learning rate.
    Now we tune the tree parameters: max_depth, min_samples_split, and min_samples_leaf
    '''


param_test2 = {'max_depth': range(5, 16, 2), 'min_samples_split': range(2, 11, 2)}
gsearch2 = GridSearchCV(
                        estimator=GradientBoostingClassifier(random_state=1, learning_rate=0.5, n_estimators=50, min_samples_leaf=1, max_features='sqrt', subsample=0.8),
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
    We got 7 as the optimal max_depth, and 2 as the optimal min_samples_split.
    Since min_samples_leaf < min_samples_split, we got 1 as the optimal min_samples_leaf as well
    Now we tune max_feature by trying 5 values from 2 to 10 in steps of 2
    '''


param_test3 = {'max_features': range(2, 11, 2)}
gsearch3 = GridSearchCV(
                        estimator=GradientBoostingClassifier(random_state=1, learning_rate=0.5, n_estimators=50, max_depth=7, min_samples_split=2, min_samples_leaf=1, subsample=0.8),
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
    We got 10 as the optimal max_features, which is the largest possible value
    Now we tune subsample
    '''


param_test4 = {'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]}
gsearch4 = GridSearchCV(
                        estimator=GradientBoostingClassifier(random_state=1, learning_rate=0.5, n_estimators=50, max_depth=7, min_samples_split=2, min_samples_leaf=1, max_features=10),
                        param_grid=param_test4, scoring='accuracy', n_jobs=-1, cv=5)
gsearch4.fit(df_x, df_y)

rint('results:')
for mean, std, params in zip(gsearch4.cv_results_['mean_test_score'], gsearch4.cv_results_['std_test_score'], gsearch4.cv_results_['params']):
    print("mean: %0.5f, std: %0.5f, params: %r" % (mean, std, params))
print('\n')

print('best parameter:')
print(gsearch4.best_params_)
print('\n')

print('best score:')
print(gsearch4.best_score_)


'''
    We got 1.0 as the optimal subsample
    '''

