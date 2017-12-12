from nn import NeuralNetwork
import numpy as np
import scipy.io as sio

data = sio.loadmat("../ExtYaleB10.mat")
# print(data["train"][0][0][:,:,0].shape)

nn = NeuralNetwork(2, 2, 1, "tanh")

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
nn.train(X, y)
for e in X:
    print(e, nn.predict(e))