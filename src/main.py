import sys
import utils
import numpy as np
from sklearn import svm
from sklearn import neighbors
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

    run_svm_linear(train['args']['classes'], dataset_train, dataset_test)
    run_svm_rbf(train['args']['classes'], dataset_train, dataset_test)
    run_knn(train['args']['classes'], dataset_train, dataset_test)

def run_svm_linear(n_class, data_train, data_test):
    print("============= SVM KERNEL LINEAR =============")
    for c in [0.1, 1, 10]:
        linear_svm = svm.SVC(C=c, kernel='linear')
        linear_svm.fit(data_train[0], data_train[1])
        predicted = linear_svm.predict(data_test[0])
        print("SETTINGS:")
        print("- C param: ", c)
        gen_metrics(gen_confusion_matrix(n_class, predicted, np.array(data_test[1])))

def run_svm_rbf(n_class, data_train, data_test):
    print("============= SVM KERNEL RBF =============")
    for c in [0.1, 1, 10]:
        for gamma in [0.1, 1, 10]:
            rbf_svm = svm.SVC(C=c, kernel='rbf', gamma=gamma)
            rbf_svm.fit(data_train[0], data_train[1])
            predicted = rbf_svm.predict(data_test[0])
            print("SETTINGS:")
            print("- C param: ", c)
            print("- gamma param: ", gamma)
            gen_metrics(gen_confusion_matrix(n_class, predicted, np.array(data_test[1])))

def run_knn(n_class, data_train, data_test):
    print("============= KNN =============")
    for k in [1, 2, 5, 10, 100]:
        knn = neighbors.KNeighborsClassifier(n_neighbors=k)
        knn.fit(data_train[0], data_train[1])
        predicted = knn.predict(data_test[0])
        print("SETTINGS:")
        print("- K param: ", k)
        gen_metrics(gen_confusion_matrix(n_class, predicted, np.array(data_test[1])))

def gen_confusion_matrix(n_class, predicted, expected):
    #tp, fp, fn, tn --> ORDEM NA MATRIX
    if n_class == 1:
        n_class = 2
    matrix = []
    for _ in range(n_class):
        matrix.append([0, 0, 0, 0])
    for i in range(len(predicted)):
        class_predicted = predicted[i]
        class_expected = expected[i]
        if class_predicted == class_expected:
            matrix[class_expected][0] += 1
            for j in range(len(matrix)):
                if j != class_expected:
                    matrix[j][3] += 1
        else:
            matrix[class_expected][2] += 1
            matrix[class_predicted][1] += 1
            for j in range(len(matrix)):
                if j != class_expected and j != class_predicted:
                    matrix[j][3] += 1

    return matrix


def gen_metrics(confusion_matrix):
    for i, analysis_class in enumerate(confusion_matrix):
        try:
            recall = round(analysis_class[0]/(analysis_class[0]+analysis_class[2])*100, 2)
        except ZeroDivisionError:
            recall = "??"
        try:
            precision = round(analysis_class[0]/(analysis_class[0]+analysis_class[1])*100, 2)
        except ZeroDivisionError :
            precision = "??"
        try:
            accuracy = round((analysis_class[0]+analysis_class[3])/(analysis_class[0]+analysis_class[1]+analysis_class[2]+analysis_class[3])*100, 2)
        except ZeroDivisionError:
            accuracy = "??"
        print("\tCLASS: ", i)
        print("\t\tRevocação: ", recall, "%")
        print("\t\tPrecisão: ", precision, "%")
        print("\t\taccuracy: ", accuracy, "%")

if __name__ == "__main__":
    main()
