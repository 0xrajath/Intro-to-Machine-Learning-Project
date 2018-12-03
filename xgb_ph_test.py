import pandas as pd
import numpy as np

import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.preprocessing import LabelEncoder
# from yellowbrick.classifier import PrecisionRecallCurve

train_loc = 'data/uci_data/poker_hand/poker-hand-training-true.data.txt'
test_loc = 'data/uci_data/poker_hand/poker-hand-testing.data.txt'
col_names = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'hand']

train = pd.read_csv(train_loc, names=col_names)
train_x = train.drop(col_names[-1], axis=1)
train_y = train[col_names[-1]]
# train_y = LabelEncoder().fit_transform(train_y)

test = pd.read_csv(test_loc, names=col_names)
test_x = test.drop(col_names[-1], axis=1)
test_y = test[col_names[-1]]
# test_y = LabelEncoder().fit_transform(test_y)

xgb_tuned = xgb.XGBClassifier(random_state=1, learning_rate=0.1, n_estimators=1600, max_depth=5, min_child_weight=3, gamma=0, reg_alpha=0.01)
# xgb_tuned = PrecisionRecallCurve(xgb_tuned, per_class=True, iso_f1_curves=True, fill_area=False, micro=False)
xgb_tuned.fit(train_x, train_y)


prediction_test = xgb_tuned.predict(test_x)
print('Testing accuracy:', accuracy_score(test_y, prediction_test))

'''
# Draw precision-recall curve
gb_tuned.score(test_x, test_y)
gb_tuned.poof()
'''
