#!/usr/bin/env python3
import numpy as np
from NeuralNetwork import NeuralNetwork
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm


def check_weight_derivatives(network, input, target, epsilon, accept_diff, print_info):
    '''
    Checks the derivatives of the network by a simple numerical differentiation method
    input/target are only one set
    accept_diff maximum of difference accepted between numerical derivative and network derivatice
    '''
    # f(x)
    output1 = network.feed_forward(inputs=input)
    loss_function1 = network.loss_function(target, output1)
    # calculate derivatives once
    network.back_propagate(network.loss_function_derivative(target, output1))
    # change the weight by epsilon (x + h)
    for i in range(len(network.layers) - 1):
        for j in range(network.layers[i]):
            for k in range(network.layers[i+1]):

                network.weights[i][j, k] += epsilon
                # f(x+h), dC/dw
                output2 = network.feed_forward(inputs=input)
                loss_function2 = network.loss_function(target, output2)
                # slope = (f(x+h) - f(x)) / h
                numerical_derivative = (loss_function2 - loss_function1) / epsilon
                difference = abs(numerical_derivative - network.weight_derivatives[i][j, k])

                if print_info:
                    print()
                    print(" Layer: {}, leftside node: {}, rightside node: {}".format(i, j, k))
                    print(" The difference is:           {:.8f}".format(difference))
                    print(" The numerical derivative is: {:.8f}".format(numerical_derivative))
                    print(" The network's derivative is: {:.8f}".format(network.weight_derivatives[i][j, k]))
                    print()
                if difference > accept_diff:
                    exit(" Derivative check failed! Check the network implementation!")
                network.weights[i][j, k] -= epsilon


def test_network(network, test_input):
    output = network.feed_forward(inputs=test_input)
    target1 = test_input[0] + test_input[1]
    target2 = test_input[0] - test_input[1]
    print()
    print(" Network: {} + {} = {:+.5f} ({:.2f})".format(test_input[0], test_input[1], output[0], target1))
    print(" Network: {} - {} = {:+.5f} ({:.2f})".format(test_input[0], test_input[1], output[1], target2))


class LossPlot():

    def __init__(self):
        fig = plt.figure()
        self.ax = fig.add_subplot(1, 1, 1)

    def plot(self, data):
        self.ax.clear()
        self.ax.set_title('Mean loss function over iterations')
        self.ax.set_xlabel('Iterations')
        self.ax.set_ylabel('Loss function')
        self.ax.set_yscale('log')
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax.plot(data)
        plt.pause(0.001)


if __name__ == "__main__":

    # For testing only one element in output is allowed
    single_input = np.array([0.3, 0.1])
    single_output = np.array([0.4])
    network = NeuralNetwork(num_inputs=2, hidden_layers=[3], num_outputs=1, activation_function='tanh')
    check_weight_derivatives(network=network, input=single_input, target=single_output, epsilon=1e-10,
                             accept_diff=1e-5, print_info=False)
    del network
    # create a Multilayer Perceptron with one hidden layer
    network = NeuralNetwork(num_inputs=2, hidden_layers=[5, 4], num_outputs=2, activation_function='tanh')

    # create a dataset to train a network for the sum operation
    inputs = np.random.rand(100, 2) * 0.5
    targets = np.stack((inputs[:, 0] + inputs[:, 1], inputs[:, 0] - inputs[:, 1]), axis=1)

    # train network
    num_iterations = 300
    # Create a generator for training and the test data
    trainer = network.train(inputs=inputs, targets=targets, epochs=num_iterations, learning_rate=0.01)
    # Train the network and plot the results
    plot = LossPlot()
    sum_errors = []
    for loss in tqdm(trainer, total=num_iterations, desc="Training neural network:"):
        sum_errors.append(loss)
        plot.plot(sum_errors)

    print()
    print("Now let's do some tests... :-)")
    # test1
    inputs = np.random.rand(100, 2) * 0.5
    test_input = np.array([0.30, 0.10])
    test_network(network, test_input)
    # test2
    test_input = np.array([0.24, 0.36])
    test_network(network, test_input)
    # test3
    test_input = np.array([0.11, 0.42])
    test_network(network, test_input)
    # test4
    test_input = np.array([0.27, 0.27])
    test_network(network, test_input)

    plt.show()
