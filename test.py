import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
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
df_y = LabelEncoder().fit_transform(df_y)
train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=0.3, random_state=0)

adab_tuned = AdaBoostClassifier(random_state=1, learning_rate=0.1, n_estimators=500,base_estimator=tree.DecisionTreeClassifier(max_depth=10))
adab_tuned = PrecisionRecallCurve(adab_tuned,per_class=True,iso_f1_curves=True,fill_area=False,micro=False)
adab_tuned.fit(train_x,train_y)

# Draw precision-recall curve
adab_tuned.score(test_x, test_y)
adab_tuned.poof()
