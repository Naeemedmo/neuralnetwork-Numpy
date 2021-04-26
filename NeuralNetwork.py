import numpy as np


class NeuralNetwork:

    def __init__(self, num_inputs, hidden_layers, num_outputs, activation_function):
        '''
        Initialize a neural network
        num_inputs number of input nodes in each training set
        hidden_layers ndarray[n,m,..]: len(ndarray) = number of hidden layers,
                                       n, m.. are number of nodes per layer
        num_outputs number of output nodes
        activation_function is "sigmoid" or "tanh"
        '''
        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs
        # choice of activation function
        if 'sigmoid' in activation_function:
            self.activation_function = self.sigmoid
            self.activation_derivative = self.sigmoid_derivative
        elif 'tanh' in activation_function:
            self.activation_function = self.hyperbolic_tangent
            self.activation_derivative = self.hyperbolic_tangent_derivative

        # an array that represent the nodes in layers
        layers = [num_inputs] + hidden_layers + [num_outputs]
        self.layers = layers

        # There is a weight matrix between each two layers with size of nodes on both side
        self.weights = [np.random.rand(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]

        # Each activation has a bias except input
        self.biases = [np.random.rand(layer) for layer in layers[1:]]

    def feed_forward(self, inputs):
        '''
        Calculate the activation for each node with the help of weights
        activation(L+1) = sigmoid(weights(L)*activation(L) + biases(L+1))
                        or   tanh(weights(L)*activation(L) + biases(L+1))
        returns activations for each node
        '''
        # There is an activation for each node of each layer
        activations = [inputs]
        for bias, weight in zip(self.biases, self.weights):
            activations.append(
                self.activation_function(np.inner(weight.T, activations[-1]).T + bias)
            )
        return activations

    def predict(self, inputs):
        '''
        return the prediction for given input
        '''
        activations = self.feed_forward(inputs)
        return activations[-1]

    def back_propagate(self, target, activations):
        '''
        Loss(target, activation) = Σ(target-activation)**2
        Apply chain rule to get derivative
        '''
        # Each bias has a derivative
        bias_derivatives = self.biases.copy()
        # There is a derivative for each weight
        weight_derivatives = self.weights.copy()
        # error = dC/da
        error = self.loss_function_derivative(target, activations[-1])
        # Walking backward to calculate weight derivatives
        for i in reversed(range(len(weight_derivatives))):
            # calculate delta dependent on activation funtion type
            # delta = dC/da * da/dz
            delta = error * self.activation_derivative(activations[i + 1])
            # dC/db = dC/da * da/dz * dx/db [note that dx/db = 1]
            bias_derivatives[i] = np.average(delta, axis=0)
            # dC/dw = dC/da * da/dz * dz/dw [note that dz/dw = a]
            weight_derivatives[i] = np.dot(activations[i].T, delta)
            # update error for next step
            # dC/da-1 = dC/da * da/dz * dz/da-1 [note that dz/da-1 = w]
            error = np.inner(self.weights[i], delta).T
        return weight_derivatives, bias_derivatives

    # update the weights and biases with derivative of loss function
    def gradient_descent(self, learning_rate, weight_derivatives, bias_derivatives):
        for i in range(len(self.weights)):
            self.weights[i] -= weight_derivatives[i] * learning_rate
        for i in range(len(self.biases)):
            self.biases[i] -= bias_derivatives[i] * learning_rate

    def train(self, inputs, targets, epochs, learning_rate, batch_size):
        for i in range(epochs):
            sum_errors = 0.0
            # shuffle the training sets first: stochastic gradient descent
            permutation = np.random.permutation(inputs.shape[0])
            shuffled_inputs = inputs[permutation]
            shuffled_targets = targets[permutation]
            n_batch = len(shuffled_inputs) // batch_size
            if n_batch * batch_size != len(shuffled_inputs):
                exit(" Input array is not dividable by batch_size! Please choose another batch_size!")
            # loop over all training sets
            for n in range(n_batch):
                j = n * batch_size
                k = (n + 1) * batch_size
                activations = self.feed_forward(shuffled_inputs[j:k])
                weight_derivatives, bias_derivatives = self.back_propagate(shuffled_targets[j:k], activations)
                self.gradient_descent(learning_rate, weight_derivatives, bias_derivatives)
                sum_errors += self.loss_function(shuffled_targets[j:k], activations[-1])
            yield sum_errors / len(shuffled_inputs)

    def loss_function(self, target, output):
        '''
        Calculate loss function
        '''
        return np.average((target - output) ** 2)

    def loss_function_derivative(self, target, output):
        '''
        Calculate loss function derivative (dC/da)
        '''
        return -2.0 * (target - output)

    # Activation functions and their derivatives
    def sigmoid(self, x):
        '''
        The function takes any real value as input and outputs values in the range 0 to 1.
        '''
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_derivative(self, sigmoid_x):
        '''
        sigmoid_x is the same as sigmoid(x)
        '''
        return sigmoid_x * (1.0 - sigmoid_x)

    def hyperbolic_tangent(self, x):
        '''
        The function takes any real value as input and outputs values in the range -1 to 1.
        '''
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def hyperbolic_tangent_derivative(self, tanh_x):
        '''
        tanh_x is the same as tanh(x)
        '''
        return 1 - tanh_x ** 2
