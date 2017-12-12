import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sb
from scipy import optimize

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def hw(w, X):
    return sigmoid(X.dot(w[:, None]))

def reg_cost(w, X, y, la):
    n = float(len(y))
    h = hw(w, X)

    t = 1e-23
    h[h < t] = t
    h[(1 - t < h) & (h < 1 + t)] = 1 - t

    reg = (float(la) / 2) * w**2
    cost = y * np.log(h) + (1 - y) * np.log(1 - h)

    return ((-sum(cost) + sum(reg)) / n)[0]

def gradient(w, X, y, la):
    n = len(y)
    h = hw(w, X)
    d_reg = float(la) * w / n
    grad = (h - y).T.dot(X) / n + d_reg.T
    return np.ndarray.flatten(grad)

def predict(w, X):
    h = hw(w, X)
    return np.round(h)

def train(X, y):
    n, d = X.shape
    w = np.zeros(d)
    la = 1
    opt_w = optimize.fmin_bfgs(reg_cost, w, args=(X, y, la), fprime=gradient)
    return opt_w

def error(w, X, y):
    p = predict(w, X)
    return 1 - np.mean(p == y)

def plot(w, X, y):
    data = pd.DataFrame({"x1": X[:,0].tolist(), "x2": X[:,1].tolist(), "y": y[:,0].tolist()})
    sb.lmplot("x1", "x2", hue="y", data=data, fit_reg=False, ci=False)

    ymin, ymax = np.min(X[:,1]), np.max(X[:,1])
    xmin, xmax = ymin * (-w[1] / w[0]), ymax * (-w[1] / w[0])
    plt.plot([xmin, xmax], [ymin, ymax])

    plt.show()


data = sio.loadmat("../HW2_Data/data2.mat")
X_train, X_test = data["X_trn"], data["X_tst"]
y_train, y_test = data["Y_trn"], data["Y_tst"]

opt_w = train(X_train, y_train)
print "Weight Vec: ", opt_w
print "Training Err: ", error(opt_w, X_train, y_train)
print "Test Err: ", error(opt_w, X_test, y_test)
plot(opt_w, X_train, y_train)
plot(opt_w, X_test, y_test)