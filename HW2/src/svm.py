import numpy as np
import random as rand
import scipy.io as sio


class SMO(object):

    def __init__(self, C=1, tol=0.1, max_passes=1000):
        self.C = C
        self.tol = tol
        self.max_passes = max_passes

    def train(self, X, y):
        # initialize
        n, d = X.shape
        a, b = np.zeros(n), 0
        passes = 0
        while passes < self.max_passes:
            n_changed_as = 0
            for i in xrange(n):
                # compute Ei
                Xi, yi = X[i,:], y[i]
                Ei = self.f(Xi, X, y, a, b) - yi
                if (yi * Ei < -self.tol and a[i] < self.C) or (yi * Ei > self.tol and a[i] > 0):
                    # compute Ej
                    j = self.get_rand(0, n - 1, i)
                    Xj, yj = X[j,:], y[j]
                    Ej = self.f(Xj, X, y, a, b) - yj
                    # save old ai, aj
                    old_ai, old_aj = a[i], a[j]
                    # compute L, H
                    if yi != yj:
                        L, H = max(0, a[j] - a[i]), min(self.C, self.C + a[j] - a[i])
                    else:
                        L, H = max(0, a[i] + a[j] - self.C), min(self.C, a[i] + a[j])
                    if L == H:
                        continue
                    # compute eta
                    eta = 2 * self.kernel(Xi, Xj) - self.kernel(Xi, Xi) - self.kernel(Xj, Xj)
                    if eta >= 0:
                        continue
                    # compute and clip new value for aj
                    a[j] -= float(yj) * (Ei - Ej) / eta
                    if a[j] > H:
                        a[j] = H
                    if a[j] < L:
                        a[j] = L
                    if abs(a[j] - old_aj) < 1e-5:
                        continue
                    # compute ai
                    a[i] += yi * yj * (old_aj - a[j])
                    # compute b1, b2
                    b1 = b - Ei - yi * (a[i] - old_ai) * self.kernel(Xi, Xi) - yj * (a[j] - old_aj) * self.kernel(Xi, Xj)
                    b2 = b - Ej - yi * (a[i] - old_ai) * self.kernel(Xi, Xj) - yj * (a[j] - old_aj) * self.kernel(Xj, Xj)
                    # compute b
                    if 0 < a[i] < self.C:
                        b = b1
                    elif 0 < a[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2.0
                    n_changed_as += 1
            if n_changed_as == 0:
                passes += 1
            else:
                passes = 0
        self.X, self.y = X, y
        self.a, self.b = a, b

    def predict(self, x):
        if self.f(x, self.X, self.y, self.a, self.b) > 0:
            return 1
        else:
            return 0

    def error(self, X, y):
        n, wrong = X.shape[0], 0
        for i in xrange(n):
            if self.predict(X[i,:]) != y[i]:
                wrong += 1
        return float(wrong) / n

    def kernel(self, x, z):
        return x.dot(z.T)

    def get_rand(self, lower, upper, exclude):
        res = exclude
        while res == exclude:
            res = rand.randint(lower, upper)
        return res

    def f(self, x, X, y, a, b):
        n, d = X.shape
        f = b
        for i in xrange(n):
            f += a[i] * y[i] * self.kernel(X[i,:], x)
        return f


data = sio.loadmat("../HW2_Data/data1.mat")
X_train, X_test = data["X_trn"], data["X_tst"]
y_train, y_test = data["Y_trn"], data["Y_tst"]

SVM = SMO()
SVM.train(X_train, y_train)
print SVM.error(X_train, y_train)
print SVM.error(X_test, y_test)