import pandas as pd
import numpy as np

import xgboost as xgb
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.preprocessing import LabelEncoder
# from yellowbrick.classifier import PrecisionRecallCurve

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

xgb_tuned = xgb.XGBClassifier(random_state=1, learning_rate=0.1, n_estimators=100, max_depth=30, min_child_weight=3, gamma=0, reg_alpha=0.01)

'''
scores = cross_val_score(xgb_tuned, train_x, train_y, cv=5, scoring='accuracy')
print(scores)
print("mean: %0.6f, std: %0.6f" % (scores.mean(), scores.std()))
'''

# xgb_tuned = PrecisionRecallCurve(xgb_tuned, per_class=True, iso_f1_curves=True, fill_area=False, micro=False)
xgb_tuned.fit(train_x, train_y)
prediction_test = xgb_tuned.predict(test_x)
print('Testing accuracy:', accuracy_score(test_y, prediction_test))


'''
gb_tuned.score(test_x, test_y)
gb_tuned.poof()
'''
