import numpy as np
from sklearn.decomposition import KernelPCA

def PCA(Y, d):
    """
    :param Y: data matrix(D * N), D: ambient dimension, N: number of points
    :param d: dimension of the subspace
    :return: mu: mean of the subspace(D)
             U: subspace basis(D * d)
             X: low-dimension representations(d * N)
    """
    mu = Y.mean(axis=1, keepdims=True)
    Y = Y - mu
    U, S, V = np.linalg.svd(Y)
    U = U[:,:d]
    X = U.T.dot(Y)
    return mu, U, X

def KPCA(K, d):
    """
    :param K: kernel matrix(N * N)
    :param d: dimension of the subspace
    :return: X: low-dimension representations(d * N)
    """
    N = len(K)
    O = np.ones((N, N)) / N
    new_K = K - K.dot(O) - O.dot(K) + O.dot(K).dot(O)
    w, v = np.linalg.eig(new_K)
    pairs = sorted([(w[i], v[:, i]) for i in range(len(w))], key=lambda x: x[0], reverse=True)
    return K.dot(np.column_stack((pairs[i][1] for i in range(d)))).T