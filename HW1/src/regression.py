import numpy as np


# Apply the non-linear transform to vector x
# args:
#   x(numpy.ndarray): input matrix: N * 1
#   n(int): n-degree polynomials [1, x,..., x^n]
# return:
#   the the new input matrix: N * (n + 1)
def non_linear_trans(x, n=1):
    N = len(x)
    newX = np.ones(N, dtype=x.dtype)
    for i in xrange(1, n + 1):
        newX = np.c_[newX, np.power(x, i)]
    return newX


# Divide data to batches of batch_size
# return: list[batch(array)]
def get_batches(data, batch_size):
    batches = []
    for i in xrange(len(data) / batch_size + 1):
        begin = i * batch_size
        end = min(len(data), (i + 1) * batch_size)
        if begin != end:
            batches.append(data[begin:end])
    return batches


# Get the i-th holdout of K-fold data
# return (holdout, remain)
def get_holdout(data, k, i):
    holdout_size = len(data) / k
    holdout = data[i * holdout_size:(i + 1) * holdout_size]
    remain = np.r_[data[:i * holdout_size], data[(i + 1) * holdout_size:]]
    return holdout, remain


# Get the error based on theta, X and corresponding y
def get_error(theta, X, y):
    return np.mean((X.dot(theta)-y)**2)


# Compute closed-form vanilla regression theta based on X and corresponding y
def vanilla_closed_form(X, y):
    return np.linalg.solve(X.T.dot(X), X.T.dot(y))


# Compute closed-form ridge regression theta based on X, corresponding y, and given lambda
def ridge_closed_form(X, y, la=0.1):
    d = X.shape[1]
    return np.linalg.solve(X.T.dot(X) + la * np.eye(d), X.T.dot(y))


# Use SGD to train vanilla regression theta
def vanilla_SGD(X, y, batch_size=1, step_size=1e-5, tolerance=1e-10):
    N, d = X.shape
    theta = np.ones(d)

    X_batches, y_batches = get_batches(X, batch_size), get_batches(y, batch_size)
    converged = False
    while not converged:
        oldTheta = np.copy(theta)
        for i in xrange(len(X_batches)):
            gradient = np.zeros(d)
            for j in xrange(len(X_batches[i])):
                gradient += (theta.T.dot(X_batches[i][j]) - y_batches[i][j]) * X_batches[i][j]
            gradient /= len(X_batches[i])
            theta -= step_size * gradient
        delta = theta - oldTheta
        if delta.T.dot(delta) < tolerance:
            converged = True
    theta.resize((d,1))
    return theta


# Use SGD to train ridge regression theta
def ridge_SGD(X, y, la=0.1, batch_size=1, step_size=1e-5, tolerance=1e-10):
    N, d = X.shape
    theta = np.ones(d)

    X_batches, y_batches = get_batches(X, batch_size), get_batches(y, batch_size)
    converged = False
    while not converged:
        oldTheta = np.copy(theta)
        for i in xrange(len(X_batches)):
            gradient = np.zeros(d)
            for j in xrange(len(X_batches[i])):
                gradient += (theta.T.dot(X_batches[i][j]) - y_batches[i][j]) * X_batches[i][j] + 2 * la * theta
            gradient /= len(X_batches[i])
            theta -= step_size * gradient
        delta = theta - oldTheta
        if delta.T.dot(delta) < tolerance:
            converged = True
    theta.resize((d,1))
    return theta


# Select the best lambda with K-fold SGD
def ridge_find_best_la(X, y, k=2, batch_size=1, step_size=1e-5, tolerance=1e-10):
    las = [1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    best_la, best_error = None, float("+inf")
    for la in las:
        error = 0
        for i in xrange(k):
            X_holdout, X_train = get_holdout(X, k, i)
            y_holdout, y_train = get_holdout(y, k ,i)
            theta = ridge_SGD(X_train, y_train, la, batch_size=batch_size, step_size=step_size, tolerance=tolerance)
            error += get_error(theta, X_holdout, y_holdout)
        if error < best_error:
            best_la, best_error = la, error
        print la, error
    return best_la