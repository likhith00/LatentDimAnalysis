from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def find_best_neighbors(X_train, y_train, X_val, y_val, neighbor_range=(5,7,9,11,15,17,20)):
    best_neighbors = None
    best_score = -1

    for n_neighbors in neighbor_range:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)

        score = accuracy_score(y_val, y_pred)

        if score > best_score:
            best_score = score
            best_neighbors = n_neighbors
    return best_neighbors, best_score


def compute_knn_score(X_train, y_train, X_test, y_test, n_neighbors):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    return score 