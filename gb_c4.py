import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from yellowbrick.classifier import PrecisionRecallCurve

data_loc = 'data/uci_data/connect_4/connect-4.data.txt'
col_names = ['pos_'+str(x) for x in range(1, 43)]
col_names.append('outcome')

df = pd.read_csv(data_loc, names=col_names)
di = {'x': 1, 'o': -1, 'b': 0, 'win': 1, 'loss': -1, 'draw': 0}
df.replace(di, inplace=True)

df_x = df.drop(col_names[-1], axis=1)
df_y = df[col_names[-1]]
# df_y = LabelEncoder().fit_transform(df_y)
train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=0.3, random_state=0)

'''
data_loc = 'data/connect_4_data/c4_game_database_blanks_removed.csv'
df = pd.read_csv(data_loc)
y_name = 'winner'

df_x = df.drop(y_name, axis=1)
df_y = df[y_name]
train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=0.3, random_state=0)
'''

'''
    There are no optimum values for learning rate as low values always work better, given that we
    train on sufficient number of trees. High learning rates can lead to overfitting. But as we reduce
    the learning rate and increase trees, the computation becomes unreasonably expensive for tuning other
    parameters.
    
    Therefore, we first choose a relatively high learning rate, and determine the optimum number
    of trees for this learning rate. Then after we fine-tune other parameters, we lower the learning rate
    and increase the trees proportionally to get more robust models.
    '''

'''
param_test1 = {'n_estimators': range(10, 101, 10)}
gsearch1 = GridSearchCV(
                        estimator=GradientBoostingClassifier(random_state=1, learning_rate=0.1, max_depth=8, min_samples_split=120, min_samples_leaf=5, max_features='sqrt', subsample=0.8),
                        param_grid=param_test1, scoring='accuracy', n_jobs=-1, cv=5)
gsearch1.fit(train_x, train_y)

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
    We got the optimal n_estimators=60 for learning_rate=1.3
    Now we tune the tree parameters: max_depth, min_samples_split, and min_samples_leaf
    '''

'''
param_test2 = {'max_depth': range(5,16,2), 'min_samples_split': range(100,901,200), 'min_samples_leaf': range(5, 26, 5)}
gsearch2 = GridSearchCV(
                        estimator=GradientBoostingClassifier(random_state=1, learning_rate=1.3, n_estimators=60, max_features='sqrt', subsample=0.8),
                        param_grid=param_test2, scoring='accuracy', n_jobs=-1, cv=5)
gsearch2.fit(train_x, train_y)

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
    We got max_depth=13, min_samples_split=500, min_samples_leaf=25
    Since min_samples_leaf=25 is the extreme value we tested, we need to try more values for it without
    changing max_depth and min_samples_split.
    '''

'''
param_test3 = {'min_samples_leaf': range(25, 46, 5)}
gsearch3 = GridSearchCV(
                        estimator=GradientBoostingClassifier(random_state=1, learning_rate=1.3, n_estimators=60, max_depth=13, min_samples_split=500, max_features='sqrt', subsample=0.8),
                        param_grid=param_test3, scoring='accuracy', n_jobs=-1, cv=5)
gsearch3.fit(train_x, train_y)

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
    We got min_samples_leaf=25
    Now we tune the randomness parameter: max_features
    '''

'''
param_test4 = {'max_features': range(5, 20, 2)}
gsearch4 = GridSearchCV(
                        estimator=GradientBoostingClassifier(random_state=1, learning_rate=1.3, n_estimators=60, max_depth=13, min_samples_split=500, min_samples_leaf=25, subsample=0.8),
                        param_grid=param_test4, scoring='accuracy', n_jobs=-1, cv=5)
gsearch4.fit(train_x, train_y)

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

'''
    We got max_features=19
    Now we tune another randomness parameter: subsample
    '''

'''
param_test5 = {'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]}
gsearch5 = GridSearchCV(
                        estimator=GradientBoostingClassifier(random_state=1, learning_rate=1.3, n_estimators=60, max_depth=13, min_samples_split=500, min_samples_leaf=25, max_features=19),
                        param_grid=param_test5, scoring='accuracy', n_jobs=-1, cv=5)
gsearch5.fit(train_x, train_y)

print('results:')
for mean, std, params in zip(gsearch5.cv_results_['mean_test_score'], gsearch5.cv_results_['std_test_score'], gsearch5.cv_results_['params']):
    print("mean: %0.5f, std: %0.5f, params: %r" % (mean, std, params))
print('\n')

print('best parameter:')
print(gsearch5.best_params_)
print('\n')

print('best score:')
print(gsearch5.best_score_)
'''

'''
    We got subsample=1.0
    '''

'''
    Now we lower the learning rate and increase the number of estimators proportionally.
    The parameter we have tuned might not be the optimum values but a good benchmark.
    '''


gb_tuned = GradientBoostingClassifier(random_state=1, learning_rate=0.156, n_estimators=500, max_depth=13, min_samples_split=500, min_samples_leaf=25, max_features=19, subsample=1.0)

scores = cross_val_score(gb_tuned, train_x, train_y, cv=5, scoring='accuracy')
print(scores)
print("mean: %0.6f, std: %0.6f" % (scores.mean(), scores.std()))

'''
# gb_tuned = PrecisionRecallCurve(gb_tuned, per_class=True, iso_f1_curves=True, fill_area=False, micro=False)
gb_tuned.fit(train_x, train_y)


prediction_train = gb_tuned.predict(train_x)
prediction_test = gb_tuned.predict(test_x)
print('Training accuracy:', accuracy_score(train_y, prediction_train))
print('Testing accuracy:', accuracy_score(test_y, prediction_test))
print('Classification report:')
print(classification_report(test_y, prediction_test))
print('Confusion matrix:')
print(confusion_matrix(test_y, prediction_test))
'''

'''
# Draw precision-recall curve
gb_tuned.score(test_x, test_y)
gb_tuned.poof()
'''
