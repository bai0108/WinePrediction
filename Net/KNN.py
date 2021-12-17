from sklearn.neighbors import KNeighborsClassifier


def KNN(n_neighbors=1):
    return KNeighborsClassifier(n_neighbors=n_neighbors)