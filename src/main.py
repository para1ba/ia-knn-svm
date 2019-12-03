import sys
import utils
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from pdb import set_trace as pause

dataset_path = '../res/dataset/'
dataset_file = sys.argv[1]

test_file = dataset_path + dataset_file + '_test.csv'
train_file = dataset_path + dataset_file + '_train.csv'

print(test_file)

def main():
    train = utils.parse_dataset(train_file)
    test = utils.parse_dataset(test_file)

    dataset_train = utils.get_data_label(train)
    dataset_test = utils.get_data_label(test)

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