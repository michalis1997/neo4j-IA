import tensorflow.python.keras.models
from matplotlib import pyplot
from tensorflow import keras
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler as ss
import pandas as pd
import warnings
import numpy as np
import statistics
from py2neo import Graph,Node,Relationship
from tensorflow.keras.layers import Dense, Dropout, Input
from keras.models import Model,Sequential
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve
import collections
# Example of a confusion matrix in Python
from sklearn.metrics import confusion_matrix,roc_auc_score


def main():

    """
    warnings.filterwarnings("ignore")
    test = pd.read_csv('C:/Users/Michalis/Desktop/CEID/5o etos/diplomatiki/intelligent agent in network/ex.csv')
    print(test)

    Test = test.values.tolist()
    print(Test)

    model_path = 'keras.h5'
    model = keras.models.load_model(model_path)

    filename = 'RandomForest.h5'
    # load the model from disk
    rf_classifier = joblib.load(filename)

    filename = 'DecisionTree.h5'
    # load the model from disk
    dt_classifier = joblib.load(filename)

    filename = 'Logistic_Regression.h5'
    # load the model from disk
    lr_classifier = joblib.load(filename)

    sc = ss()

    test = sc.fit_transform(Test)
    predictions = (model.predict(test) > 0.5).astype(int)

    for i in range(len(test)):
        print(f"\t Logistic Regression: {lr_classifier.predict([test[i]])}")
        print(f"\t Decision Tree: {dt_classifier.predict([test[i]])}")
        print(f"\t Random Forest: {rf_classifier.predict([test[i]])}")
        print(f"\t keras: {predictions[i]}")
    """
    metrics0 = pd.read_csv('C:/Users/Michalis/Desktop/CEID/5o etos/diplomatiki/intelligent agent in network/out0.csv')
    metrics1 = pd.read_csv('C:/Users/Michalis/Desktop/CEID/5o etos/diplomatiki/intelligent agent in network/out1.csv')
    metrics2 = pd.read_csv('C:/Users/Michalis/Desktop/CEID/5o etos/diplomatiki/intelligent agent in network/out2.csv')

    Train = metrics0[['Label','adamic_score','common_neighbors','preferential_attachment',
                   'resource_allocation','totalNeighbors']]

    Train1 = metrics1[['Label','adamic_score','common_neighbors','preferential_attachment',
                   'resource_allocation','totalNeighbors']]

    Train2 = metrics2[['Label', 'adamic_score', 'common_neighbors', 'preferential_attachment',
                       'resource_allocation', 'totalNeighbors']]

    Train = Train.append(Train1, ignore_index=True)
    Train = Train.append(Train2, ignore_index=True)

    metrics0 = metrics0[['adamic_score','common_neighbors','preferential_attachment',
                   'resource_allocation','totalNeighbors']]

    R1 = []
    R2 = []
    R3 = []
    R1 = metrics1.loc[(metrics1['StartNode'] == 2213) | (metrics1['StartNode'] == 130261) | (metrics1['StartNode'] == 130262)]

    R1 = R1[['adamic_score', 'common_neighbors', 'preferential_attachment',
                         'resource_allocation', 'totalNeighbors']]

    harmonic_mean = []
    singular_value = []

    u, s, vh = np.linalg.svd(metrics0, full_matrices=False)
    mean = statistics.harmonic_mean(s)
    singular_value.append(s)
    harmonic_mean.append(mean)

    u, s, vh = np.linalg.svd(R1, full_matrices=False)
    mean = statistics.harmonic_mean(s)
    singular_value.append(s)
    harmonic_mean.append(mean)

    print(singular_value)
    print(harmonic_mean)

    output = [sigmoid(x) for x in harmonic_mean]
    print(output)

    T = Train['Label'].values.tolist()
    LEN = T.count(1)
    NS_AUC = 0.500000
    AUC = [NS_AUC,neural_network(Train), neural_network1(Train), neural_network2(Train), neural_network3(Train)]


def neural_network(Train):

    # data preprocessing
    X = Train.iloc[:,Train.columns != "Label"].values
    Y = Train.iloc[:,  0].values
    counter = collections.Counter(Y)
    print(counter)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=1)

    model = Sequential()

    model.add(Dense(10,activation="selu",input_dim=5))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    model.fit(X_train,y_train,validation_data=(X_test, y_test),epochs=25,batch_size=512)
    score = model.evaluate(X_test,y_test)
    print(score)

    # predict crisp classes for test set
    y_predict = model.predict(X_test)
    y_predict = (y_predict > 0.5)
    list(y_predict)
    results = confusion_matrix(y_test,y_predict)
    print(results)

    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(y_test))]

    # ROC AUC
    ns_auc = roc_auc_score(y_test,ns_probs)
    auc = roc_auc_score(y_test, y_predict)
    print('No Skill:ROC AUC: %f' % ns_auc)
    print('ROC AUC: %f' % auc)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test, y_predict)
    print('F1 score: %f' % f1)

    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, y_predict)

    filename = "neural1.h5"
    model = model.save(filename)

    return auc


def neural_network1(Train):

    # data preprocessing
    X = Train.iloc[:, Train.columns != "Label"].values
    Y = Train.iloc[:, 0].values
    counter = collections.Counter(Y)
    print(counter)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)

    model = Sequential()

    model.add(Dense(25, activation="relu", input_dim=5))
    model.add(Dense(1, activation="sigmoid"))

    predictions = model(X_train[:1]).numpy()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=25,batch_size=512)
    score = model.evaluate(X_test, y_test)
    print(score)
    # predict crisp classes for test set
    y_predict = model.predict(X_test)
    y_predict = (y_predict > 0.5)
    list(y_predict)

    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(y_test))]

    ns_auc = roc_auc_score(y_test, ns_probs)
    auc = roc_auc_score(y_test, y_predict)
    print('No Skill:ROC AUC: %f' % ns_auc)
    print('ROC AUC: %f' % auc)

    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test,y_predict)
    print('F1 score: %f' % f1)

    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, y_predict)

    results = confusion_matrix(y_test, y_predict)
    print(results)

    filename = "neural2.h5"
    model = model.save(filename)

    return auc


def neural_network2(Train):

    # data preprocessing
    X = Train.iloc[:, Train.columns != "Label"].values
    Y = Train.iloc[:, 0].values
    counter = collections.Counter(Y)
    print(counter)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)

    model = Sequential()

    model.add(Dense(10, activation="relu", input_dim=5))
    model.add(Dense(10, activation="selu"))
    model.add(Dense(1, activation="sigmoid"))

    predictions = model(X_train[:1]).numpy()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25,batch_size=512)
    score = model.evaluate(X_test, y_test)
    print(score)

    # predict crisp classes for test set
    y_predict = model.predict(X_test)

    y_predict = (y_predict > 0.5)
    list(y_predict)

    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(y_test))]

    ns_auc = roc_auc_score(y_test, ns_probs)
    auc = roc_auc_score(y_test, y_predict)
    print('No Skill:ROC AUC: %f' % ns_auc)
    auc = roc_auc_score(y_test, y_predict)
    print('ROC AUC: %f' % auc)

    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test, y_predict)
    print('F1 score: %f' % f1)

    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, y_predict)

    results = confusion_matrix(y_test, y_predict)
    print(results)

    filename = "neural3.h5"
    model = model.save(filename)

    return auc


def neural_network3(Train):

    # data preprocessing
    X = Train.iloc[:, Train.columns != "Label"].values
    Y = Train.iloc[:, 0].values
    counter = collections.Counter(Y)
    print(counter)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)

    model = Sequential()

    model.add(Dense(25, activation="relu", input_dim=5))
    model.add(Dense(25, activation="selu"))
    model.add(Dense(1, activation="sigmoid"))

    predictions = model(X_train[:1]).numpy()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25,batch_size=512)
    score = model.evaluate(X_test, y_test)
    print(score)

    # predict crisp classes for test set
    y_predict = model.predict(X_test)

    y_predict = (y_predict > 0.5)
    list(y_predict)

    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(y_test))]

    ns_auc = roc_auc_score(y_test, ns_probs)
    auc = roc_auc_score(y_test, y_predict)
    print('No Skill:ROC AUC: %f' % ns_auc)
    auc = roc_auc_score(y_test, y_predict)
    print('ROC AUC: %f' % auc)

    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test, y_predict)
    print('F1 score: %f' % f1)

    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, y_predict)

    results = confusion_matrix(y_test, y_predict)
    print(results)

    filename = "neural4.h5"
    model = model.save(filename)

    return auc


# sigmoid activation function
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


# Rectified Linear Unit (ReLU)
def ReLU(x):
    data = [max(0,value) for value in x]
    return np.array(data, dtype=float)


if __name__ == "__main__":
    main()

