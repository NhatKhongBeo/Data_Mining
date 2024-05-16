import numpy as np
import copy


class NeuralNetwork:
    def __init__(self, layers_dims):
        self.layers_dims = layers_dims
        self.parameters = self.initialize_parameters()
        self.costs_iter = []
        self.costs_epoch = []
        self.valid_cost = []
        self.batch_size = 32

    # sigmoid function
    def sigmoid(self, Z):
        A = 1 / (1 + np.exp(-Z))
        cache = Z
        return A, cache

    # relu function
    def relu(self, Z):
        A = np.maximum(0, Z)
        assert A.shape == Z.shape
        cache = Z
        return A, cache

    # create He parameters
    def initialize_parameters(self):
        np.random.seed(3)
        parameters = {}

        L = len(self.layers_dims)

        for l in range(1, L):
            # parameters['W' + str(l)] = np.random.randn(self.layers_dims[l], self.layers_dims[l - 1]) * 0.01
            parameters["W" + str(l)] = np.random.randn(
                self.layers_dims[l], self.layers_dims[l - 1]
            ) * np.sqrt(2.0 / self.layers_dims[l - 1])
            parameters["b" + str(l)] = np.zeros((self.layers_dims[l], 1))
            assert parameters["W" + str(l)].shape == (
                self.layers_dims[l],
                self.layers_dims[l - 1],
            )
            assert parameters["b" + str(l)].shape == (self.layers_dims[l], 1)
        return parameters

    # sigmoid backward function
    def sigmoid_backward(self, dA, cache):
        Z = cache
        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)
        assert dZ.shape == Z.shape
        return dZ

    # relu backward function
    def relu_backward(self, dA, cache):
        Z = cache
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        assert dZ.shape == Z.shape
        return dZ

    # forward propagation
    # linear forward function
    def linear_forward(self, A, W, b):
        Z = np.dot(W, A) + b
        assert Z.shape == (W.shape[0], A.shape[1])
        cache = (A, W, b)
        return Z, cache

    # linear activation forward function
    def linear_activation_forward(self, A_prev, W, b, activation):
        Z, linear_cache = self.linear_forward(A_prev, W, b)
        if activation == "sigmoid":
            A, activation_cache = self.sigmoid(Z)
        elif activation == "relu":
            A, activation_cache = self.relu(Z)
        assert A.shape == (W.shape[0], A_prev.shape[1])
        cache = (linear_cache, activation_cache)
        return A, cache

    # forward propagation
    def forward_propagation(self, X, parameters):
        caches = []
        A = X
        L = len(parameters) // 2
        for l in range(1, L):
            A_prev = A
            A, cache = self.linear_activation_forward(
                A_prev,
                parameters["W" + str(l)],
                parameters["b" + str(l)],
                activation="relu",
            )
            caches.append(cache)
        AL, cache = self.linear_activation_forward(
            A, parameters["W" + str(L)], parameters["b" + str(L)], activation="sigmoid"
        )
        caches.append(cache)

        return AL, caches

    # forward propagation with dropout
    def forward_propagation_with_dropout(self, X, parameters, keep_prob):
        caches = []
        A = X
        L = len(parameters) // 2
        for l in range(1, L):
            A_prev = A
            A, cache = self.linear_activation_forward(
                A_prev,
                parameters["W" + str(l)],
                parameters["b" + str(l)],
                activation="relu",
            )
            D = np.random.rand(A.shape[0], A.shape[1])
            D = (D < keep_prob).astype(int)
            A = A * D
            A = A / keep_prob
            cache = (cache, D)
            caches.append(cache)
        AL, cache = self.linear_activation_forward(
            A, parameters["W" + str(L)], parameters["b" + str(L)], activation="sigmoid"
        )
        caches.append(cache)

        return AL, caches

    # compute cost
    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
        cost = np.squeeze(cost)
        assert cost.shape == ()
        return cost

    def compute_cost_with_regularization(self, AL, Y, parameters, lambd):
        m = Y.shape[1]
        L = len(parameters) // 2
        cross_entropy_cost = self.compute_cost(AL, Y)
        L2_regularization_cost = 0
        for l in range(1, L + 1):
            L2_regularization_cost += np.sum(np.square(parameters["W" + str(l)]))
        L2_regularization_cost = (lambd / (2 * m)) * L2_regularization_cost
        cost = cross_entropy_cost + L2_regularization_cost
        return cost

    # backward propagation
    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = 1 / m * np.dot(dZ, A_prev.T)
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        assert dA_prev.shape == A_prev.shape
        assert dW.shape == W.shape
        assert db.shape == b.shape

        return dA_prev, dW, db

    # linear activation backward function
    def linear_activation_backward(self, dA, cache, activation):
        linear_cache, activation_cache = cache
        if activation == "relu":
            dZ = self.relu_backward(dA, activation_cache)
        elif activation == "sigmoid":
            dZ = self.sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        return dA_prev, dW, db

    # backward propagation
    def backward_propagation(self, AL, Y, caches):
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        current_cache = caches[L - 1]
        dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(
            dAL, current_cache, activation="sigmoid"
        )
        grads["dA" + str(L - 1)] = dA_prev_temp
        grads["dW" + str(L)] = dW_temp
        grads["db" + str(L)] = db_temp

        for l in reversed(range(L - 1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(
                grads["dA" + str(l + 1)], current_cache, activation="relu"
            )
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    # backward propagation with regularization
    def backward_propagation_with_regularization(self, AL, Y, caches, lambd):
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        current_cache = caches[L - 1]
        dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(
            dAL, current_cache, activation="sigmoid"
        )
        grads["dA" + str(L - 1)] = dA_prev_temp
        grads["dW" + str(L)] = dW_temp + (lambd / m) * self.parameters["W" + str(L)]
        grads["db" + str(L)] = db_temp

        for l in reversed(range(L - 1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(
                grads["dA" + str(l + 1)], current_cache, activation="relu"
            )
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = (
                dW_temp + (lambd / m) * self.parameters["W" + str(l + 1)]
            )
            grads["db" + str(l + 1)] = db_temp

        return grads

    # backward propagation with dropout
    def backward_propagation_with_dropout(self, AL, Y, caches, keep_prob):
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        dAL = AL - Y

        current_cache = caches[L - 1]
        dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(
            dAL, current_cache, activation="sigmoid"
        )
        grads["dA" + str(L - 1)] = dA_prev_temp
        grads["dW" + str(L)] = dW_temp
        grads["db" + str(L)] = db_temp

        for l in reversed(range(L - 1)):
            current_cache = caches[l]
            cache_tmp, D = current_cache
            dA = grads["dA" + str(l + 1)]
            dA = dA * D
            dA = dA / keep_prob
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(
                dA, cache_tmp, activation="relu"
            )
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    # update parameters
    def update_parameters(self, parameters, grads, learning_rate):
        parameters = copy.deepcopy(parameters)
        L = len(parameters) // 2
        for l in range(L):
            parameters["W" + str(l + 1)] = (
                parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
            )
            parameters["b" + str(l + 1)] = (
                parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
            )
        return parameters

    # fit iterations function
    # def fit(self,X,Y,num_iterations,learning_rate,lambd=0.7,print_cost=False):
    #     np.random.seed(1)
    #     costs = []
    #     for i in range(0,num_iterations):
    #         AL,caches = self.forward_propagation(X,self.parameters)
    #         #cost = self.compute_cost_with_regularization(AL,Y)
    #         cost = self.compute_cost_with_regularization(AL,Y,self.parameters,lambd)
    #         grads = self.backward_propagation_with_regularization(AL,Y,caches,lambd)
    #         self.parameters = self.update_parameters(self.parameters,grads,learning_rate)
    #         if print_cost and i % 100 == 0 or i == num_iterations-1:
    #             print ("Cost after iteration %i: %f" %(i, cost))
    #             costs.append(cost)
    #     self.costs_iter = costs

    def fit(
        self,
        X,
        Y,
        num_iterations,
        learning_rate=0.01,
        lambd=0,
        keep_prob=1,
        print_cost=False,
    ):
        np.random.seed(1)
        costs = []
        for i in range(0, num_iterations):
            if keep_prob == 1:
                AL, caches = self.forward_propagation(X, self.parameters)
            elif keep_prob < 1:
                AL, caches = self.forward_propagation_with_dropout(
                    X, self.parameters, keep_prob
                )

            if lambd == 0:
                cost = self.compute_cost(AL, Y)
            else:
                cost = self.compute_cost_with_regularization(
                    AL, Y, self.parameters, lambd
                )

            # cost = self.compute_cost_with_regularization(AL,Y,self.parameters,lambd)

            if lambd == 0 and keep_prob == 1:
                grads = self.backward_propagation(AL, Y, caches)
            elif lambd != 0:
                grads = self.backward_propagation_with_regularization(
                    AL, Y, caches, lambd
                )
            elif keep_prob < 1:
                grads = self.backward_propagation_with_dropout(AL, Y, caches, keep_prob)
            self.parameters = self.update_parameters(
                self.parameters, grads, learning_rate
            )
            if print_cost and i % 10000 == 0:
                print("Cost after iteration {}: {}".format(i, cost))
            if print_cost and i % 1000 == 0:
                costs.append(cost)
        self.costs_iter = costs

    # fit epochs function
    # def train(self,X,Y,num_epochs,learning_rate,batch_size,lambd=0.7, print_cost=False):
    #     np.random.seed(1)
    #     costs = []

    #     m= X.shape[1]

    #     for epoch in range(num_epochs):
    #         epoch_cost = 0
    #         num_interations = m // batch_size

    #         for i in range(num_interations):
    #             start = i * batch_size
    #             end = min(start + self.batch_size, m)
    #             X_batch = X[:,start:end]
    #             Y_batch = Y[:,start:end]

    #             AL,caches = self.forward_propagation(X_batch,self.parameters)

    #             cost = cost = self.compute_cost_with_regularization(AL,Y_batch,self.parameters,lambd)

    #             epoch_cost += cost/num_interations

    #             grads = self.backward_propagation_with_regularization(AL,Y_batch,caches,lambd)
    #             self.parameters = self.update_parameters(self.parameters,grads,learning_rate)

    #         costs.append(epoch_cost)
    #         if print_cost:
    #             print(f"Cost after epoch {epoch}: {epoch_cost}")

    #     self.costs_epoch = costs

    def train(
        self,
        X,
        Y,
        X_val,
        Y_val,
        num_epochs,
        learning_rate=0.01,
        batch_size=32,
        lambd=0,
        keep_prob=1,
        print_cost=False,
    ):
        np.random.seed(1)
        costs = []

        m = X.shape[1]

        for epoch in range(num_epochs):
            epoch_cost = 0
            num_interations = m // batch_size

            for i in range(num_interations):
                start = i * batch_size
                end = min(start + self.batch_size, m)
                X_batch = X[:, start:end]
                Y_batch = Y[:, start:end]

                if keep_prob == 1:
                    AL, caches = self.forward_propagation(X_batch, self.parameters)
                elif keep_prob < 1:
                    AL, caches = self.forward_propagation_with_dropout(
                        X_batch, self.parameters, keep_prob
                    )

                if lambd == 0:
                    cost = self.compute_cost(AL, Y_batch)
                else:
                    cost = self.compute_cost_with_regularization(
                        AL, Y_batch, self.parameters, lambd
                    )

                # cost = self.compute_cost_with_regularization(AL,Y,self.parameters,lambd)

                if lambd == 0 and keep_prob == 1:
                    grads = self.backward_propagation(AL, Y_batch, caches)
                elif lambd != 0:
                    grads = self.backward_propagation_with_regularization(
                        AL, Y_batch, caches, lambd
                    )
                elif keep_prob < 1:
                    grads = self.backward_propagation_with_dropout(
                        AL, Y_batch, caches, keep_prob
                    )
                self.parameters = self.update_parameters(
                    self.parameters, grads, learning_rate
                )
                epoch_cost += cost / num_interations

            costs.append(epoch_cost)
            if print_cost:
                if X_val is not None and Y_val is not None:
                    AL_val, _ = self.forward_propagation(X_val, self.parameters)
                    cost_val = self.compute_cost(AL_val, Y_val)
                    print(
                        f"Cost after epoch {epoch}: {epoch_cost}, Cost validation: {cost_val}"
                    )
                    self.predict(X_val, Y_val)
                else:
                    print(f"Cost after epoch {epoch}: {epoch_cost}")

        self.costs_epoch = costs

    def predict(self, X, y):
        m = X.shape[1]
        p = np.zeros((1, m))
        probas, caches = self.forward_propagation(X, self.parameters)

        for i in range(0, probas.shape[1]):
            if probas[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0
        print("Accuracy: " + str(np.sum((p == y) / m)))
        return p
