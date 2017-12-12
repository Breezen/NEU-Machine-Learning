import numpy as np

def dist(a, b, axis=1):
    return np.linalg.norm(a - b, axis=axis)

def getRandomCentroids(X, k):
    idx = np.random.choice(np.arange(X.shape[0]), size=k, replace=False)
    return X[idx]

def getLabels(X, centroids):
    labels = []
    for i, p in enumerate(X):
        dists = dist(p, centroids)
        labels.append(np.argmin(dists))
    return labels

def getCentroids(X, labels, k):
    centroids = []
    for i in range(k):
        points = [X[j] for j in range(len(X)) if labels[j] == i]
        centroids.append(np.mean(points, axis=0))
    return centroids

def getError(X, centroids, labels):
    error = 0
    for i, p in enumerate(X):
        error += dist(p, centroids[labels[i]], axis=0)
    return error

def KMeans(X, k, r):
    """
    :param X: data matrix(D * N)
    :param k: desired number of clusters
    :param r: number of repetitions with different random initializations
    :return: centroids: cluster centroids of the data(k * D)
             labels: clustering labels of the data(N)
    """
    X = X.T
    min_error, best_centroids, best_labels = float("inf"), None, None
    for i in range(r):
        labels = np.zeros(len(X))
        centroids = getRandomCentroids(X, k)
        while True:
            old_labels = np.copy(labels)
            labels = getLabels(X, centroids)
            if np.array_equal(labels, old_labels):
                error = getError(X, centroids, labels)
                if error < min_error:
                    min_error, best_centroids, best_labels = error, centroids, labels
                break
            centroids = getCentroids(X, labels, k)
    return best_centroids, best_labels