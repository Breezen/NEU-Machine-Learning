import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from pca import PCA, KPCA
from kmeans import KMeans

data = sio.loadmat("../HW3_Data/dataset1.mat")
Y = data["Y"]
mu, U, X = PCA(Y, 2)

centroids40, labels40 = KMeans(Y, 2, 10)
centroids2, labels2 = KMeans(X, 2, 10)

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
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    fig, ax = plt.subplots()
    for i in range(2):
        points = np.array([X.T[j] for j in range(len(X.T)) if labels40[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=16, c=colors[i])
    # centroids40_2 = PCA(np.array(centroids40).T, 2)[-1]
    # ax.scatter(centroids40_2[0], centroids40_2[1], marker='*', s=200, c='#050505')
    plt.show()

def plot5():
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    fig, ax = plt.subplots()
    for i in range(2):
        points = np.array([X.T[j] for j in range(len(X.T)) if labels2[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=16, c=colors[i])
    centroids2_t = np.array(centroids2)
    ax.scatter(centroids2_t[:, 0], centroids2_t[:, 1], marker='*', s=200, c='#050505')
    plt.show()

plot1()
plot2()
plot3()
plot4()
plot5()