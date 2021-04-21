#!/usr/bin/env python3
import numpy as np
from NeuralNetwork import NeuralNetwork

def check_weight_derivatives(network, input, target, epsilon):
    '''
    Checks the derivatives of the network by a simple numerical differentiation method
    input/target are only one set
    '''
    #indexes for the address of derivative we check
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
                # f(x+h)
                output2 = network.feed_forward(inputs=input)
                loss_function2 = network.loss_function(target, output2)
                # slope = (f(x+h) - f(x)) / h
                numerical_derivative = (loss_function2 - loss_function1) / epsilon
                difference = abs(numerical_derivative - network.weight_derivatives[i][j, k])

                if difference > 1e-5:
                    print()
                    print(" The difference is:           {:.8f}".format(difference))
                    print(" The numerical derivative is: {:.8f}".format(numerical_derivative))
                    print(" The network's derivative is: {:.8f}".format(network.weight_derivatives[i][j, k]))
                # undo changes
                network.weights[i][j, k] -= epsilon




if __name__ == "__main__":

    # create a dataset to train a network for the sum operation
    inputs = np.random.rand(1000,2) * 0.5
    targets = np.stack((inputs[:,0] + inputs[:,1], inputs[:,0] - inputs[:,1]), axis=1)

    # For testing only one element in output is allowed
    #targets = inputs[:,0] + inputs[:,1]
    #network = NeuralNetwork(num_inputs=2, hidden_layers=[5], num_outputs=1)
    #test_network(network=network, input=inputs[0], target=targets[0], epsilon=1e-10)
    #exit()
    # create a Multilayer Perceptron with one hidden layer
    network = NeuralNetwork(num_inputs=2, hidden_layers=[5], num_outputs= 2)

    #exit()
    # train network
    network.train(inputs=inputs, targets=targets, epochs=1000, learning_rate=0.01)

    # create dummy data
    input = np.array([0.3, 0.1])
    target = np.array([0.4, 0.2])

    # get a prediction
    output = network.feed_forward(inputs=input)

    print()
    print("Our network believes that {} + {} is equal to {:.8f}".format(input[0], input[1], output[0]))
    print("Our network believes that {} - {} is equal to {:.8f}".format(input[0], input[1], output[1]))

