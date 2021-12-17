import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    filePath = "../data/winequality-white.csv"
    trainPath = "../data/white_train2.csv"
    testPath = "../data/white_test2.csv"
    trainSetRatio = 0.8

    df = pd.read_csv(filePath, sep=';')
    x = df.drop(columns=['quality']).copy()
    df.loc[(df['quality'] == 3)] = 0
    df.loc[(df['quality'] == 4)] = 0
    df.loc[(df['quality'] == 5)] = 0
    df.loc[(df['quality'] == 6)] = 0
    df.loc[(df['quality'] == 7)] = 1
    df.loc[(df['quality'] == 8)] = 1
    df.loc[(df['quality'] == 9)] = 1

    y = df['quality']
    x['quality'] = y
    # print(x)
    # x.to_csv("../data/whiteWine_3labels.csv", index=False)

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, train_size=trainSetRatio, test_size=1-trainSetRatio)
    X_train['quality'] = Y_train
    X_test['quality'] = Y_test
    # print(X_train)
    X_train.to_csv(trainPath, index=False)
    X_test.to_csv(testPath, index=False)



