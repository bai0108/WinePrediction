import pandas as pd
from imblearn.under_sampling import *


def get_data(filename):
    train = pd.read_csv(filename, sep=',')
    x_train = train.drop(columns=['quality']).copy()
    y_train = train['quality'].to_frame()
    return x_train, y_train


def get_dataset(train_path, test_path, removed_fields=[]):
    train = pd.read_csv(train_path, sep=',')
    test = pd.read_csv(test_path, sep=',')
    for f in removed_fields:
        train = train.drop(columns=[f])
        test = test.drop(columns=[f])
    x_train = train.drop(columns=['quality']).copy()
    y_train = train['quality'].to_frame()

    x_test = test.drop(columns=['quality']).copy()
    y_test = test['quality'].to_frame()
    return x_train, y_train, x_test, y_test


def get_distribution(df):
    result = {}
    for _, row in df.iterrows():
        label = row['quality']
        if label in result:
            result[label] += 1
        else:
            result[label] = 1
    return result


def imbalanced_learning(x, y):
    cc = AllKNN(n_neighbors=1)
    x_res, y_res = cc.fit_resample(x, y)
    print(y.shape, y_res.shape)
    print(get_distribution(y))
    print(get_distribution(y_res))
    return x_res, y_res


def label_distribution(train_path, test_path):
    x_, train_label, _, test_label = get_dataset(train_path, test_path)
    train = get_distribution(train_label)
    test = get_distribution(test_label)
    print(train, test)


if __name__ == '__main__':
    ''' drop duplicates '''
    # data = pd.read_csv("../data/whiteWine_3labels.csv")
    # data = data.drop_duplicates()
    # data.to_csv("../data/whiteWine3.csv", index=False)
    # print(data)

    # data = pd.read_csv("../data/whiteWine1.csv")
    # data = data.drop(columns=['pH', 'density'])
    # data.to_csv("../data/whiteWine2.csv", index=False)
    # print(data)
