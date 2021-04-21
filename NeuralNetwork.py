
import numpy as np

#TODO: biases to be included

class NeuralNetwork:

    def __init__(self, num_inputs, hidden_layers, num_outputs):
        '''
        Initialize a neural network

        Input Arguments:
        num_inputs number of input nodes in each training set
        hidden_layers an array with n elements, each element is presenting one hiddenlayer with element nodes
        num_outputs number of output nodes

        '''
        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        # an array that represent the nodes in layers
        layers = [num_inputs] + hidden_layers + [num_outputs]
        self.layers = layers

        # There is a weight matrix between each two layers with size of nodes on both side
        weights = []
        for i in range(len(layers) -1):
            w = np.random.rand(layers[i], layers[i+1])
            weights.append(w)
        self.weights = weights
        # There is a derivative for each weight
        weight_derivatives = []
        for i in range(len(layers)-1):
            d = np.zeros((layers[i], layers[i+1]))
            weight_derivatives.append(d)
        self.weight_derivatives = weight_derivatives
        # There is an activation for each node of each layer
        activation = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activation.append(a)
        self.activation = activation
        # Each activation has a bias except input
        biases = []
        for i in range(1,len(layers)):
            b = np.random.rand(layers[i])
            biases.append(b)
        self.biases = biases
        # Each bias has a derivative
        bias_derivatives = []
        for i in range(1,len(layers)):
            db = np.zeros(layers[i])
            bias_derivatives.append(db)
        self.bias_derivatives = bias_derivatives

    def feed_forward(self, inputs):
        '''
        Calculate the activation for each node with the help of weights
        activation(L+1) = sigmoid(weights(L)*activation(L))
        returns activations for each node
        '''
        self.activation[0] = inputs
        for i, weight in enumerate(self.weights):
            self.activation[i+1] = self.sigmoid(np.inner(weight.T, self.activation[i]) + self.biases[i])
        return self.activation[-1]

    def back_propagate(self, loss_function_derivative):
        '''
        Loss(target, activation) = Σ(target-activation)**2
        error = 2 * (target-activation)
        Apply chain rule to get derivitive
        '''
        error = loss_function_derivative # = dC/da
        # Walking backward to calculate weight derivatives
        for i in reversed(range(len(self.weight_derivatives))):
            # delta = dC/da * da/dz
            delta = error * self.sigmoid_derivative(self.activation[i+1])
            # dC/db = dC/da * da/dz * dx/db [note that dx/db = 1]
            self.bias_derivatives[i] = delta
            # dC/dw = dC/da * da/dz * dz/dw [note that dz/dw = a]
            self.weight_derivatives[i] = np.outer(self.activation[i], delta)
            # update error for next step
            # dC/da-1 = dC/da * da/dz * dz/da-1 [note that dz/da-1 = w]
            error = np.inner(self.weights[i], delta)

    # update the weights and biases with derivative of loss function
    def gradient_descent(self, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= self.weight_derivatives[i] * learning_rate
        for i in range(1,len(self.biases)):
            self.biases[i] -= self.bias_derivatives[i] * learning_rate


    def train(self, inputs, targets, epochs, learning_rate):

        for i in range(epochs):
            sum_errors = 0.0
            # loop over all training sets
            for input, target in zip(inputs, targets):
                output = self.feed_forward(input)
                self.back_propagate(self.loss_function_derivative(target, output))
                self.gradient_descent(learning_rate)
                sum_errors += self.loss_function(target, output)

            # report mean error of all training sets each 10th epoch
            print("Error: {:.10f} at epoch {}".format(sum_errors / len(inputs), i+1)) if i%10 == 0 else None


    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_derivative(self, sigmoid_x):
        '''
        sigmoid_x is the same as sigmoid(x)
        '''
        return sigmoid_x * (1.0 - sigmoid_x)

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


