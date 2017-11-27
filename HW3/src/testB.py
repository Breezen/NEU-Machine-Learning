import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from pca import PCA, KPCA
from kmeans import KMeans
from spectral_clustering import spectral_clustering

data = sio.loadmat("../HW3_Data/dataset2.mat")
Y = data["Y"]
mu, U, X = PCA(Y, 2)

plt.style.use("seaborn")

def plot1():
    plt.scatter(Y[0], Y[1], s=16)
    plt.show()

def plot2():
    plt.scatter(Y[1], Y[2], s=16)
    plt.show()

def plot3():
    plt.scatter(X[0], X[1], s=16)
    plt.show()

def plot4():
    centroids, labels = KMeans(X, 2, 10)
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    fig, ax = plt.subplots()
    for i in range(2):
        points = np.array([X.T[j] for j in range(len(X.T)) if labels[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=16, c=colors[i])
    centroids_t = np.array(centroids)
    ax.scatter(centroids_t[:, 0], centroids_t[:, 1], marker='*', s=200, c='#050505')
    plt.show()

def get_rbf_kernel(Y):
    N = Y.shape[1]
    kernel = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            kernel[i, j] = np.sum((Y[:, i] - Y[:, j]) ** 2) / 8
    return np.exp(kernel)

def plot5():
    K = get_rbf_kernel(Y)
    KX = KPCA(K, 2)
    centroids, labels = KMeans(KX, 2, 10)
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    fig, ax = plt.subplots()
    for i in range(2):
        points = np.array([KX.T[j] for j in range(len(KX.T)) if labels[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=16, c=colors[i])
    centroids_t = np.array(centroids)
    ax.scatter(centroids_t[:, 0], centroids_t[:, 1], marker='*', s=200, c='#050505')
    plt.show()

    fig, ax = plt.subplots()
    for i in range(2):
        points = np.array([X.T[j] for j in range(len(X.T)) if labels[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=16, c=colors[i])
    plt.show()


plot1()
plot2()
plot3()
plot4()
plot5()