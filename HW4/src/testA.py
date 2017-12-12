from nn import NeuralNetwork
import numpy as np
import scipy.io as sio

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

# scale
X_scale, y_scale = np.amax(X_train, axis=0), 10
X_train = X_train / X_scale # maximum of X array
y_train = y_train / y_scale # max tag is 10

print(X_train.shape)
print(y_train.shape)

nn = NeuralNetwork(32256, 100, 1, "tanh")
nn.train(X_train, y_train, epochs=10000)
for img in X_train:
    print(nn.predict(img))