import numpy as np
from typing import Iterator
from activation_function import ActivationFunction


class NeuralNetwork:
    """ Multi layer perceptron neural network

    """
    def __init__(self,
                 num_inputs: int,
                 hidden_layers: list[int],
                 num_outputs: int,
                 activation_function: ActivationFunction()
                 ):
        """ Initializes a neural network

        Args:
            num_inputs: number of input neurons
            hidden_layers: number of neurons in each hidden layer
            num_outputs: number of output neurons
            activation_function: :math:`\\sigma(x)` See ActivationFunction class description
        """
        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs
        # choice of activation function
        self.activation_function = activation_function

        # an array that represent the neurons in layers
        layers = [num_inputs] + hidden_layers + [num_outputs]
        self.layers = layers

        # There is a weight matrix between each two layers with size of neurons on both side
        self.weights = [np.random.rand(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]

        # Each activation has a bias except input
        self.biases = [np.random.rand(layer) for layer in layers[1:]]

    def feed_forward(self, inputs: np.ndarray) -> list[np.ndarray]:
        """ Calculates the activation for each neurons in the whole network:

        :math:`a_{i} = \\sigma(a_{i-1}.w_i + b_i)`

        where :math:`\\sigma` is the activation function and :math:`a_{i}` is the activations
        for layer i, :math:`w_i` and :math:`b_i` are weights and biases between layer i-1 and i

        Args:
            inputs: an m-by-n matrix, where m is the batch size and n is the number of input
            neurons

        Returns:
            a list of m-by-n matrices, where m is the batch size and n is the number of neurons of
            each layer
        """
        # There is an activation for each neurons of each layer
        activations = [np.atleast_2d(inputs)]
        for bias, weight in zip(self.biases, self.weights):
            activations.append(
                self.activation_function.evaluate(np.inner(weight.T, activations[-1]).T + bias)
            )
        return activations

    def predict(self, inputs: np.ndarray) -> list[np.ndarray]:
        """ Predicts the output (:math:`a_{i}`) for a given input (:math:`a_{0}`)

        Args:
            inputs: an m-by-n matrix, where m is the batch size and n is the number of input
            neurons

        Returns:
            the prediction (activations of the last layer), which is an m-by-n matrix, where m is
            the batch size and n is the number of output neurons
        """
        activations = self.feed_forward(inputs)
        return activations[-1]

    def back_propagate(self,
                       target: np.ndarray,
                       activations: list[np.ndarray]
                       ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """ Calculates the weights and biases for the network

        Args:
            target: an m-by-n matrix, where m is the batch size and n is the number of output
            neurons
            activations: returned by :meth:`feed_forward`

        Returns:
            a tuple containing two 2D numpy arrays

                - :math:`\\frac{\\partial L}{\\partial w}` \
                (:math:`L` loss function, :math:`w` weights)

                - :math:`\\frac{\\partial L}{\\partial b}` \
                (:math:`L` loss function, :math:`b` biases)
        """
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
            delta = error * self.activation_function.derivative(activations[i + 1])
            # dC/db = dC/da * da/dz * dx/db [note that dx/db = 1]
            bias_derivatives[i] = np.average(delta, axis=0)
            # dC/dw = dC/da * da/dz * dz/dw [note that dz/dw = a]
            weight_derivatives[i] = np.dot(activations[i].T, delta)
            # update error for next step
            # dC/da-1 = dC/da * da/dz * dz/da-1 [note that dz/da-1 = w]
            error = np.inner(self.weights[i], delta).T
        return weight_derivatives, bias_derivatives

    def gradient_descent(self,
                         learning_rate: float,
                         weight_derivatives: list[np.ndarray],
                         bias_derivatives: list[np.ndarray]
                         ):
        """ Updates the weights and biases with derivative of loss function

        Args:
            learning_rate: learning rate of the gradient descent
            weight_derivatives: :math:`\\frac{\\partial L}{\\partial w}`
            bias_derivatives: :math:`\\frac{\\partial L}{\\partial b}`

        """
        for i in range(len(self.weights)):
            self.weights[i] -= weight_derivatives[i] * learning_rate
        for i in range(len(self.biases)):
            self.biases[i] -= bias_derivatives[i] * learning_rate

    def train(self,
              inputs: np.ndarray,
              targets: np.ndarray,
              epochs: int,
              learning_rate: float,
              batch_size: int
              ) -> Iterator[float]:
        """ Trains the neural network

            Args:
                inputs: an m-by-n matrix, where m is number of traning sets and n is number of
                inputs per set
                targets: an m-by-n matrix, where m is number of training sets and n is number of
                outputs per set
                epochs: number of iterations for training
                learning_rate: learning rate of the gradient descent
                batch_size: number of training sets for each traning \n
                            batch_size = 1 (stochastic gradient descent) \n
                            batch_size < traning set (mini-batch gradient descent) \n
                            batch_size = traning set (batch gradient descent)
            Yields:
                mean loss function value per epoch

        """
        for i in range(epochs):
            sum_errors = 0.0
            # shuffle the training sets first: stochastic gradient descent
            permutation = np.random.permutation(inputs.shape[0])
            shuffled_inputs = inputs[permutation]
            shuffled_targets = targets[permutation]
            n_batch = len(shuffled_inputs) // batch_size
            if n_batch * batch_size != len(shuffled_inputs):
                exit(" Input array is not dividable by batch_size!\
                    Please choose another batch_size!")
            # loop over all training sets
            for n in range(n_batch):
                j = n * batch_size
                k = (n + 1) * batch_size
                activations = self.feed_forward(shuffled_inputs[j:k])
                weight_derivatives, bias_derivatives = self.back_propagate(
                    shuffled_targets[j:k], activations
                    )
                self.gradient_descent(learning_rate, weight_derivatives, bias_derivatives)
                sum_errors += self.loss_function(shuffled_targets[j:k], activations[-1])
            yield sum_errors / len(shuffled_inputs)

    def loss_function(self,
                      target: np.ndarray,
                      output: np.ndarray
                      ) -> float:
        """ Calculates loss function :math:`L = \\frac{1}{n}\\sum_{i=1}^{n}(y_i-\\hat y_i)^2`

        Args:
            target: target values, an m-by-n matrix, where m is the batch size and n is the number
            of output neurons (:math:`y_i`)
            output: network predicted values, an m-by-n matrix, where m is the batch size and n is
            the number of output neurons (:math:`\\hat y_i`)

        Returns:
            Averaged loss function value for targets and outputs
        """
        return np.average((target - output) ** 2)

    def loss_function_derivative(self,
                                 target: np.ndarray,
                                 output: np.ndarray
                                 ) -> np.ndarray:
        """ Calculates loss function derivative
        :math:`\\frac{\\partial L}{\\partial \\hat y} = -2.0\\times(y_i-\\hat y_i)`

        Args:
            target: target values, an m-by-n matrix, where m is the batch size and
            n is the number of output neurons (:math:`y_i`)
            output: network predicted values, an m-by-n matrix, where m is the batch
            size and n is the number of output neurons (:math:`\\hat y_i`)

        Returns:
            An m-by-n matrix of loss function derivatives for target and output

        """
        return -2.0 * (target - output)
