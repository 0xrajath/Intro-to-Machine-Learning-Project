import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from yellowbrick.classifier import PrecisionRecallCurve

train_loc = 'data/uci_data/poker_hand/poker-hand-training-true.data.txt'
test_loc = 'data/uci_data/poker_hand/poker-hand-testing.data.txt'
col_names = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'hand']

train = pd.read_csv(train_loc, names=col_names)
train_x = train.drop(col_names[-1], axis=1)
train_y = train[col_names[-1]]
train_y = LabelEncoder().fit_transform(train_y)

test = pd.read_csv(test_loc, names=col_names)
test_x = test.drop(col_names[-1], axis=1)
test_y = test[col_names[-1]]
test_y = LabelEncoder().fit_transform(test_y)

gb_tuned = GradientBoostingClassifier(random_state=1, learning_rate=0.05, n_estimators=500, max_depth=7, min_samples_split=2, min_samples_leaf=1, max_features=10, subsample=1)
gb_tuned = PrecisionRecallCurve(gb_tuned, per_class=True, iso_f1_curves=True, fill_area=False, micro=False)
gb_tuned.fit(train_x, train_y)

'''
prediction_train = gb_tuned.predict(train_x)
prediction_test = gb_tuned.predict(test_x)
print('Training accuracy:', accuracy_score(train_y, prediction_train))
print('Testing accuracy:', accuracy_score(test_y, prediction_test))
print('Classification report:')
print(classification_report(test_y, prediction_test))
print('Confusion matrix:')
print(confusion_matrix(test_y, prediction_test))
'''

# Draw precision-recall curve
gb_tuned.score(test_x, test_y)
gb_tuned.poof()
