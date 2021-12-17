import numpy as np
import sklearn.metrics as metrics
from Net.MyNet import MyNet
from utils.util import get_dataset

''' get predict data'''

binary_trainFile_w = 'data/white_train_binary.csv'
binary_testFile_w = 'data/white_test_binary.csv'
binary_trainFile_r = 'data/red_train_binary.csv'
binary_testFile_r = 'data/red_test_binary.csv'

multi_trainFile_r = 'data/red_train_multi.csv'
multi_testFile_r = 'data/red_test_multi.csv'
multi_trainFile_w = 'data/white_train_multi.csv'
multi_testFile_w = 'data/white_test_multi.csv'

if __name__ == '__main__':
    model_type = "mul"   # mul / binary
    wine = "white"  # input Red / white
    if model_type == 'binary':
        if wine == 'Red':
            _, _, x_test, y_true = get_dataset(binary_trainFile_r, binary_testFile_r)
            x_test = np.array(x_test)
            model = MyNet(11, 2)
            model.load_weights("Model/red_weight_binary.h5")
        else:
            _, _, x_test, y_true = get_dataset(binary_trainFile_w, binary_testFile_w)
            x_test = np.array(x_test)
            model = MyNet(11, 2)
            model.load_weights("Model/white_weight_binary.h5")
    else:
        if wine == 'Red':
            _, _, x_test, y_true = get_dataset(multi_trainFile_r, multi_testFile_r)
            x_test = np.array(x_test)
            model = MyNet(11, 6)
            model.load_weights("Model/red_weight_multi.h5")
        else:
            _, _, x_test, y_true = get_dataset(multi_trainFile_w, multi_testFile_w)
            x_test = np.array(x_test)
            model = MyNet(11, 7)
            model.load_weights("Model/white_weight_multi.h5")

    '''classification'''
    res = model.predict(x_test)
    y_pred = [np.argmax(i) for i in res]
    f1_each = metrics.f1_score(y_true, y_pred, average=None)
    f_score = metrics.f1_score(y_true, y_pred, average='weighted')
    accuracy = metrics.accuracy_score(y_true, y_pred)
    confusion_metrix = metrics.confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(confusion_metrix)
    print(f"Accuracy: {accuracy}")
    print(f"f1-score: {f_score}")
    print(f1_each)
    # {1: 3627, 2: 140, 0: 151} {0: 32, 1: 908, 2: 40}
