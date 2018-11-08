import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

#Read the training and testing data set
data_train = pd.read_csv(filepath_or_buffer="poker-hand-training-true.data", sep=',', header=None)
data_test = pd.read_csv(filepath_or_buffer="poker-hand-testing.data", sep=',', header=None)

#Print shape of data sets
print(data_train.shape)
print(data_test.shape)

#Ready the Train Data
array_train = data_train.values
data_train = array_train[:,0:10]
label_train = array_train[:,10]
#Ready the Test Data
array_test = data_test.values
data_test = array_test[:,0:10]
label_test = array_test[:,10]

models = [BaggingClassifier(),RandomForestClassifier(),AdaBoostClassifier]
model_names = ["Bagging","Random Forest", "AdaBoost"]

for model,name in zip(models,model_names):
    model.fit(data_train,label_train)

    if name == "Random Forest":
        print(model.feature_importances_)
        #predict
    prediction = model.predict(data_test)
        #Print accuracy
    acc = accuracy_score(label_test, prediction)
    print("Accuracy Using", name, ": " + str(acc) + '\n')
    print(classification_report(label_test, prediction))
    print(confusion_matrix(label_test, prediction))