import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - x**2

def softplus(x):
    return np.log(1.0 + np.exp(x))

class AutoEncoder:

    def __init__(self, s1, s2, activation="sigmoid"):
        self.activation = {
            "sigmoid": sigmoid,
            "tanh": tanh,
            "softplus": softplus
        }[activation]

        self.activation_prime = {
            "sigmoid": sigmoid_prime,
            "tanh": tanh_prime,
            "softplus": sigmoid
        }[activation]

        # set weights
        self.weights = [1e-5 * np.random.randn(s1 + 1, s2), 1e-5 * np.random.randn(s2, s1)]

    def train(self, X, learning_rate=0.2, epochs=100000):
        # add bias unit to input layer
        X = np.c_[X, np.ones(len(X))]

        for k in range(epochs):
            i = np.random.randint(len(X))
            a = [X[i]]
            for l in range(len(self.weights)):
                dot = np.dot(a[l], self.weights[l])
                activation = self.activation(dot)
                a.append(activation)

            # output layer
            error = X[i] - a[-1]
            deltas = [error * self.activation_prime(a[-1])]

            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_prime(a[l]))
            deltas.reverse()

            # back propagation
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

        return self.weights[0]