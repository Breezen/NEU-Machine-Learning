from pca import PCA
from kmeans import KMeans
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt

def flatImageByClass(data, tag):
    X, y = [], []
    h, w, n = data.shape
    for i in range(n):
        X.append(data[:, :, i].reshape(h * w))
        y.append(tag)
    return X, y

data = sio.loadmat("../ExtYaleB10.mat")
data_train, data_test = data["train"], data["test"]

X_train, y_train = [], []
for tag in range(10):
    X_add, y_add = flatImageByClass(data_train[0][tag], tag)
    X_train += X_add
    y_train += y_add
X_train, y_train = np.array(X_train), np.array(y_train)

X_test, y_test = [], []
for tag in range(10):
    X_add, y_add = flatImageByClass(data_test[0][tag], tag)
    X_test += X_add
    y_test += y_add
X_test, y_test = np.array(X_test), np.array(y_test)

# KMeans k = 10
cents, labels = KMeans(X_train.T, 10, 10)
# Error
error = 0
for i in range(len(labels)):
    if labels[i] != y_train[i]:
        error += 1
print("Error ratio: ", float(error) / len(labels))

# PCA d = 2
X_train = PCA(X_train.T, 2)[-1].T

# Plot
plt.style.use("seaborn")
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'burlywood', 'chartreuse']
fig, ax = plt.subplots()
for i in range(10):
    points = np.array([X_train[j] for j in range(len(X_train)) if labels[j] == i])
    ax.scatter(points[:, 0], points[:, 1], s=16, c=colors[i])
plt.show()