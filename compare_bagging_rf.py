import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier


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
    RandomForestClassifier(random_state=1, n_estimators=10, max_features=0.9),
    RandomForestClassifier(random_state=1, n_estimators=10, max_features=0.7),
    RandomForestClassifier(random_state=1, n_estimators=10, max_features=0.5),
    RandomForestClassifier(random_state=1, n_estimators=10, max_features=0.3),
    RandomForestClassifier(random_state=1, n_estimators=10, max_features=0.1),
    RandomForestClassifier(random_state=1, n_estimators=10, max_features=1)
]

avg_bias_squared = []
avg_var = []

n_repeat = 50
n_train = len(train_x)
n_test = len(test_x)
group_size = int(n_train/n_repeat)

print("data size:", len(df_x))
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
