import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier


def main():
    dataset = []
    data, labels = [], []
    run_svm_linear(data, labels)
    run_svm_rbf(data, labels)
    run_knn(data, labels)

def run_svm_linear(data, labels):
    linear_svm = svm.SVC(kernel='linear')
    for c in [0.1, 1, 10]:
        pass

def run_svm_rbf(data, labels):
    rbf_svm = svm.SVC(kernel='rbf')
    for c in [0.1, 1, 10]:
        for gamma in [0.1, 1, 10]:
            pass

def run_knn(data, labels):
    for k in [1, 2, 5, 10, 100]:
        pass

if __name__ == "__main__":
    main()