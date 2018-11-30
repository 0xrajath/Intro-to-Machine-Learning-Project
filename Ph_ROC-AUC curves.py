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

# Read the data into a Pandas dataframe
poker_df = pd.read_csv("poker_hand_train.csv")

# Label the columns and classes based on the dataset description from the UCI Repository
poker_df.columns = ['first_suit', 'first_rank', 'second_suit', 'second_rank', 'third_suit', 'third_rank',
                     'fourth_suit', 'fourth_rank', 'fifth_suit', 'fifth_rank', 'hand']
classes = ['Nothing', 'one_pair', 'two_pair', 'three_of_a_kind', 'straight', 'flush', 'full_house',
            'four_of_a_kind', 'straight_flush', 'royal_flush']

# Separate the data into features (X) and targets (y)
X = poker_df.iloc[:,0:9]
y = poker_df['hand']


# Combine all classes above flush into one class termed 'flush or better'
poker_df.loc[poker_df['hand'] >= 5, 'hand'] = 5
y = poker_df['hand']
classes = ['Nothing', 'one_pair', 'two_pair', 'three_of_a_kind', 'straight', 'flush_or_better']

#Split the data into training and testing data set
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3)


clf = GradientBoostingClassifier(random_state=1, max_depth=10, n_estimators=100, learning_rate=0.1)
# clf = AdaBoostClassifier(random_state=1,base_estimator=tree.DecisionTreeClassifier(max_depth=10), n_estimators=100,
#                           learning_rate=0.1)
clf.fit(X_train, y_train)


#Generate RUC-AUC curve for classifier
rocauc = ROCAUC(clf, size=(1080, 720), classes=classes)

rocauc.score(X_test, y_test)
r = rocauc.poof()

# Generate classification report for the given classifier
# report = ClassificationReport(clf, size=(1080, 720), classes=classes)
#
# report.score(X_test, y_test)
# c = report.poof()

#Generate Prediction error for each class
# error = ClassPredictionError(clf, size=(1080, 720), classes=classes)
#
# error.score(X_test, y_test)
# e = error.poof()



#Additional generation of accuracy scores, classification report, and confusion matrix
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