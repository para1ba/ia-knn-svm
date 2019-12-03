import numpy as np
from sklearn import svm
from sklearn import neighbors


def main():
    dataset = []
    data, labels = [], []
    run_svm_linear(data, labels)
    run_svm_rbf(data, labels)
    run_knn(data, labels)

def run_svm_linear(data_train, data_test):
    for c in [0.1, 1, 10]:
        linear_svm = svm.SVC(C=c, kernel='linear')
        linear_svm.fit(data_train[0], data_train[1])
        predicted = linear_svm.predict(data_test[0])
        gen_metrics(gen_confusion_matrix(predicted, data_test[1]))

def run_svm_rbf(data_train, data_test):
    for c in [0.1, 1, 10]:
        for gamma in [0.1, 1, 10]:
            rbf_svm = svm.SVC(C=c, kernel='rbf', gamma=gamma)
            rbf_svm.fit(data_train[0], data_train[1])
            predicted = rbf_svm.predict(data_test[0])
            gen_metrics(gen_confusion_matrix(predicted, data_test[1]))

def run_knn(data_train, data_test):
    for k in [1, 2, 5, 10, 100]:
        knn = neighbors.KNeighborsClassifier(n_neighbors=k)
        knn.fit(data_train[0], data_train[1])
        predicted = knn.predict(data_test[0])
        gen_metrics(gen_confusion_matrix(predicted, data_test[1]))

def gen_confusion_matrix(predicted, labels):
    #tp, fp, fn, tn = 0, 0, 0, 0
    matrix = []
    for _ in range(len(labels[0])):
        matrix.append((0, 0, 0, 0))
    


def gen_metrics(confusion_matrix):
    pass

if __name__ == "__main__":
    main()