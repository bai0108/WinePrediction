from Train.SVM_and_KNN import *
from utils.util import get_dataset

if __name__ == '__main__':
    train_path = 'data/red_train_binary.csv'
    test_path = 'data/red_test_binary.csv'
    removed_fields = []
    model_name = 'nn'
    use_imbalanced = False

    im_n_neighbors = 1 # AllKNN
    n_neighbors = 1 # knn
    class_weight={0: 1, 1: 1} # svm

    x_train, y_train, x_test, y_test = get_dataset(train_path, test_path, removed_fields=removed_fields)

    model = None
    if model_name == 'svm':
        model = SVM(class_weight)
        train_model(x_train, y_train, x_test, y_test, model, use_imbalanced, im_n_neighbors)
    else:
        model = KNN(n_neighbors)
        train_model(x_train, y_train, x_test, y_test, model, use_imbalanced, im_n_neighbors)
