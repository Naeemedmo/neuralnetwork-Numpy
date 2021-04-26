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
    activations = network.feed_forward(inputs=input)
    loss_function1 = network.loss_function(target, activations[-1])
    # calculate derivatives once
    weight_derivatives, bias_derivatives = network.back_propagate(target, activations)
    # change the weight by epsilon (x + h)
    for i in range(len(network.layers) - 1):
        for j in range(network.layers[i]):
            for k in range(network.layers[i+1]):
                network.weights[i][j, k] += epsilon
                # f(x+h), dC/dw
                output2 = network.predict(inputs=input)
                loss_function2 = network.loss_function(target, output2)
                # slope = (f(x+h) - f(x)) / h
                numerical_derivative = (loss_function2 - loss_function1) / epsilon
                difference = abs(numerical_derivative - weight_derivatives[i][j, k])

                if print_info:
                    print()
                    print(" Layer: {}, leftside node: {}, rightside node: {}".format(i, j, k))
                    print(" The difference is:           {:.8f}".format(difference))
                    print(" The numerical derivative is: {:.8f}".format(numerical_derivative))
                    print(" The network's derivative is: {:.8f}".format(weight_derivatives[i][j, k]))
                    print()
                if difference > accept_diff:
                    exit(" Derivative check failed! Check the network implementation!")
                network.weights[i][j, k] -= epsilon


def test_network(network, test_input):
    output = network.predict(inputs=test_input)
    target = [test_input[0] + test_input[1], test_input[0] - test_input[1]]
    print()
    print(" Network: {} + {} = {:+.5f} ({:.2f})".format(test_input[0], test_input[1], output[0], target[0]))
    print(" Network: {} - {} = {:+.5f} ({:.2f})".format(test_input[0], test_input[1], output[1], target[1]))
    print(" Network: Loss function = {:.7f}".format(network.loss_function(target=target, output=output)))


class LossPlot():

    def __init__(self):
        fig = plt.figure()
        self.ax = fig.add_subplot(1, 1, 1)

    def plot(self, data, test_data):
        self.ax.clear()
        self.ax.set_title('Mean loss function over iterations')
        self.ax.set_xlabel('Iterations')
        self.ax.set_ylabel('Loss function')
        self.ax.set_yscale('log')
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax.plot(data, label='training error')
        self.ax.plot(test_data, label='testing error')
        self.ax.legend()
        plt.pause(0.001)


if __name__ == "__main__":

    # For testing only one element in output is allowed
    single_input = np.array([0.3, 0.1])
    single_output = np.array([0.4])
    network = NeuralNetwork(num_inputs=2, hidden_layers=[3], num_outputs=1, activation_function='tanh')
    check_weight_derivatives(network=network, input=single_input, target=single_output, epsilon=1e-10,
                             accept_diff=1e-5, print_info=False)
    # create a Multilayer Perceptron with one hidden layer
    network = NeuralNetwork(num_inputs=2, hidden_layers=[4, 4], num_outputs=2, activation_function='tanh')

    # create a dataset to train a network for the sum operation
    inputs = np.random.rand(5000, 2) * 0.5
    targets = np.stack((inputs[:, 0] + inputs[:, 1], inputs[:, 0] - inputs[:, 1]), axis=1)

    # train network
    num_iterations = 100
    # Create a generator for training
    trainer = network.train(inputs=inputs, targets=targets, epochs=num_iterations,
                            learning_rate=0.01, batch_size=100)
    # test data for live plot
    test_inputs = np.random.rand(100, 2) * 0.5
    test_targets = np.stack((test_inputs[:, 0] + test_inputs[:, 1],
                             test_inputs[:, 0] - test_inputs[:, 1]), axis=1)

    # Train the network and plot the results
    plot = LossPlot()
    training_loss = []
    test_loss = []
    for loss in tqdm(trainer, total=num_iterations, desc="Training neural network:"):
        test_outputs = network.predict(test_inputs)
        test_loss.append(network.loss_function(test_targets, test_outputs))
        training_loss.append(loss)
        plot.plot(training_loss, test_loss)

    print()
    print(" Now let's do some tests... :-)")
    test_network(network, [0.30, 0.10])
    test_network(network, [0.24, 0.36])
    test_network(network, [0.11, 0.42])
    test_network(network, [0.27, 0.27])

    plt.show()
