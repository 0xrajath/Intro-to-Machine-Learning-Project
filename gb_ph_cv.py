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
    We lower the learning rate and increase the number of estimators proportionally.
    The parameter we have tuned might not be the optimum values but a good benchmark.
    '''

gb_tuned = GradientBoostingClassifier(random_state=1, learning_rate=0.05, n_estimators=500, max_depth=7, min_samples_split=2, min_samples_leaf=1, max_features=10, subsample=1)
scores = cross_val_score(gb_tuned, df_x, df_y, cv=5, scoring='accuracy')
print(scores)
print("mean: %0.6f, std: %0.6f" % (scores.mean(), scores.std()))
