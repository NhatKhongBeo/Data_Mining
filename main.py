from Model import *
import numpy as np
np.random.seed(0)
X = np.random.randn(200, 2)
X = X.reshape(X.shape[0],-1).T
print(X)
Y = np.random.randint(0, 2, (1, 200))
n_x = X.shape[0]
print(n_x)
layer_dims = [n_x,20,7,5,1]
model = NeuralNetwork(layer_dims)

model.fit(X, Y, learning_rate = 0.0075, num_iterations = 3000, print_cost = True)

print(model.parameters)

