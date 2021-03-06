import numpy as np
import scipy.io as sio
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

clf = LogisticRegression()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))