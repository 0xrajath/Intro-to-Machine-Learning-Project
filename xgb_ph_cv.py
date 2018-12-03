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
    We lower the learning rate and increase the number of estimators proportionally.
    The parameter we have tuned might not be the optimum values but a good benchmark.
    '''

xgb_tuned = xgb.XGBClassifier(random_state=1, learning_rate=0.1, n_estimators=1600, max_depth=5, min_child_weight=3, gamma=0, reg_alpha=0.01)
scores = cross_val_score(xgb_tuned, df_x, df_y, cv=5, scoring='accuracy')
print(scores)
print("mean: %0.6f, std: %0.6f" % (scores.mean(), scores.std()))

