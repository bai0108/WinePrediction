import csv
import random


# def get_data(filename):
#     """
#     filename : The path of dataset
#     """
#     features = []
#     labels = []
#     with open(filename, newline='') as file:
#         reader = csv.reader(file)
#         next(reader)
#         rows = [rows for rows in reader]
#         # random.shuffle(rows)
#
#     for row in rows:
#         for items in row:
#             items = list(map(float, items.split(";")))
#             features.append(items[:-1])
#             labels.append(items[-1])
#
#     labels = list(map(int, labels))
#     return features, labels


if __name__ == '__main__':
    # Path = '../data/winequality-red.csv'
    # feature, label = get_data(Path)
    #
    # for i in feature[:5]:
    #     print(i)
    # print(label)
    print('aa: %d:%s' % (1, 'bai'))

