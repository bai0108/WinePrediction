from utils.util import get_dataset


def get_distribution(df):
    result = {}
    for _, row in df.iterrows():
        label = row['quality']
        if label in result:
            result[label] += 1
        else:
            result[label] = 1
    return result


def label_distribution(train_path, test_path):
    _, train_label, _, test_label = get_dataset(train_path, test_path)
    train = get_distribution(train_label)
    test = get_distribution(test_label)
    print(train, test)


if __name__ == '__main__':
    label_distribution('../data/white_train2.csv', '../data/white_test2.csv')
