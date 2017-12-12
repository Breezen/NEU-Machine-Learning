from pca import PCA
import numpy as np
import scipy.io as sio
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

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

# PCA
X_train = PCA(X_train.T, 100)[-1].T
X_test = PCA(X_test.T, 100)[-1].T
print(X_train.shape)
print(X_test.shape)

# SVM
clf2 = LinearSVC()
clf2.fit(X_train, y_train)
print("SVM accuracy: ", clf2.score(X_test, y_test))

# Logistic Regression
clf3 = LogisticRegression()
clf3.fit(X_train, y_train)
print("LR accuracy: ", clf3.score(X_test, y_test))