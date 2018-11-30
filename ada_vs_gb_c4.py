import pandas as pd
import matplotlib.pyplot as plt

from xgboost.sklearn import XGBClassifier
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split as tts
from yellowbrick.classifier import ClassBalance, ROCAUC, ClassificationReport, ClassPredictionError
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

poker_df = pd.read_csv("c4_game_database_blanks_removed.csv")

# Separate the data into features (X) and targets (y)
X = poker_df.iloc[:,0:42]
y = poker_df['winner']

#Split into training and testing sets
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3)

#Fit Gradient Boost and Ada Boost algorithms with similar parameters
clf = GradientBoostingClassifier(random_state=1, max_depth=10, n_estimators=100, learning_rate=0.1)
# clf = AdaBoostClassifier(random_state=1,base_estimator=tree.DecisionTreeClassifier(max_depth=10), n_estimators=100,
#                          learning_rate=0.1)
clf.fit(X_train, y_train)


accuracy_train = []
accuracy_test = []

prediction = clf.predict(X_test)
prediction_train = clf.predict(X_train)

accuracy_test.append(accuracy_score(y_test, prediction))
accuracy_train.append(accuracy_score(y_train, prediction_train))

print(clf)
print('Testing accuracy: ', accuracy_test)
print('Training accuracy: ', accuracy_train)
print('Classification Report: ')
print(classification_report(y_test, prediction))
print('Confusion Matrix: ')
print(confusion_matrix(y_test, prediction))
print('\n')
print('\n')