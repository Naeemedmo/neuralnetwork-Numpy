#!/usr/bin/env python3
import numpy as np
from neuralnetwork import NeuralNetwork


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


if __name__ == "__main__":

    # test num_inputs num_outputs

    # test batch_size

    # test activation functions tanh sigmoid

    # For testing only one element in output is allowed
    single_input = np.array([0.3, 0.1])
    single_output = np.array([0.4])
    network = NeuralNetwork(num_inputs=2, hidden_layers=[3], num_outputs=1, activation_function='tanh')
    check_weight_derivatives(network=network, input=single_input, target=single_output, epsilon=1e-10,
                             accept_diff=1e-5, print_info=False)
