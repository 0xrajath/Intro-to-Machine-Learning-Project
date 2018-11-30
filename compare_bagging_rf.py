import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

from timeit import default_timer as timer


#df = pd.read_csv("data/poker_hand_train_resample4.csv")
df = pd.read_csv("data/poker_hand_train.csv")
#df = pd.read_csv("data/c4_game_database.csv")
#df = df.fillna(0)

label_name = 'hand'
#label_name = 'winner'
df_x = df.drop(label_name, axis=1)
df_y = df[label_name]
train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=0.3, random_state=0)

N_MODELS = 7
model_names = ['Bagging (1)', 'Random Forests (.9)', 'Random Forests (.7)', 'Random Forests (.5)', 'Random Forests (.3)', 'Random Forests (.1)', 'Pure Random Forests']
models = [
    BaggingClassifier(random_state=1, n_estimators=10),
    RandomForestClassifier(random_state=1, n_estimators=100, max_features=0.9),
    RandomForestClassifier(random_state=1, n_estimators=100, max_features=0.7),
    RandomForestClassifier(random_state=1, n_estimators=100, max_features=0.5),
    RandomForestClassifier(random_state=1, n_estimators=100, max_features=0.3),
    RandomForestClassifier(random_state=1, n_estimators=100, max_features=0.1),
    RandomForestClassifier(random_state=1, n_estimators=100, max_features=1)
]

'''
accuracy_train = []
accuracy_test = []
train_time = []

for i in range(N_MODELS):
    start_time = timer()
    models[i].fit(train_x, train_y)
    time_used = timer() - start_time
    prediction_test = models[i].predict(test_x)
    prediction_train = models[i].predict(train_x)

    train_time.append(time_used)
    accuracy_test.append(accuracy_score(test_y, prediction_test))
    accuracy_train.append(accuracy_score(train_y, prediction_train))
    
    print(model_names[i])
    print('Training time:', train_time[i])
    print('Training accuracy:', accuracy_train[i])
    print('Testing accuracy:', accuracy_test[i])
    print('Classification report:')
    print(classification_report(test_y, prediction_test))
    print('Confusion matrix:')
    print(confusion_matrix(test_y, prediction_test))
    print('\n\n')

for i in range(N_MODELS):
    print('Training time for', model_names[i], ':', train_time[i])
print('\n')

for i in range(N_MODELS):
    print('Training accuracy for', model_names[i], ':', accuracy_train[i])
print('\n')

for i in range(N_MODELS):
    print('Testing accuracy for', model_names[i], ':', accuracy_test[i])
'''


avg_bias_squared = []
avg_var = []

n_repeat = 30
n_train = len(train_x)
n_test = len(test_x)
group_size = int(n_train/n_repeat)

print("data size:", len(df_x))
#print("training size:", group_size)
print("training size:", n_train - group_size)
print("testing size:", n_test)
print('\n')


for i in range(N_MODELS):
    predictions = np.zeros((n_test, n_repeat))
    for j in range(n_repeat):
        sub_train_x = train_x.drop(train_x.index[[j*group_size, (j+1)*group_size]])
        sub_train_y = train_y.drop(train_y.index[[j*group_size, (j+1)*group_size]])
        #sub_train_x = train_x[j*group_size:(j+1)*group_size]
        #sub_train_y = train_y[j*group_size:(j+1)*group_size]
        models[i].fit(sub_train_x, sub_train_y)
        print(accuracy_score(test_y, models[i].predict(test_x)))
        predictions[:, j] = models[i].predict(test_x)
    bias_squared = (np.mean(predictions, axis=1) - test_y)**2
    var = np.var(predictions, axis=1)
    avg_bias_squared.append(np.sum(bias_squared)/n_test)
    avg_var.append(np.sum(var)/n_test)
    print(model_names[i])
    print("Average bias squared:", avg_bias_squared[i])
    print("Average variance:", avg_var[i])
    print('\n')

for i in range(N_MODELS):
    print("Average bias squared for", model_names[i], ":", avg_bias_squared[i])
    print("Average variance for", model_names[i], ":", avg_var[i])
