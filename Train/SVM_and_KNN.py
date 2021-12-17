from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from utils.analysis import get_distribution
from imblearn.under_sampling import *


def KNN(n_neighbors=1):
    return KNeighborsClassifier(n_neighbors=n_neighbors)


def SVM(class_weight={0: 1, 1: 1}):
    return SVC(class_weight=class_weight)


def imbalanced_learning(x, y, n_neighbors=1):
    cc = AllKNN(n_neighbors=n_neighbors)
    x_res, y_res = cc.fit_resample(x, y)
    print(y.shape, y_res.shape)
    print(get_distribution(y))
    print(get_distribution(y_res))
    return x_res, y_res


def train_model(x_train, y_train, x_test, y_test, model, use_imbalanced=False, im_n_neighbors=1):
    sc = StandardScaler()
    if use_imbalanced:
        x_train, y_train = imbalanced_learning(x_train, y_train, im_n_neighbors)
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)
    model.fit(x_train, y_train)
    pred_labels = model.predict(x_test)
    acc = accuracy_score(y_test, pred_labels)
    f1 = f1_score(y_test, pred_labels, average=None)
    f11 = f1_score(y_test, pred_labels, average='weighted')
    print(acc, f1, f11)


if __name__ == '__main__':
    train_path = 'data/red_train2.csv'
    test_path = 'data/red_test2.csv'
    # train_path = 'data/white_train2.csv'
    # test_path = 'data/white_test2.csv'
    # x_train, y_train, x_test, y_test = get_dataset(train_path, test_path)
    # print(get_distribution(y_test))
    # print(get_distribution(y_train))
    # train_model(x_train, y_train, x_test, y_test, KNN())