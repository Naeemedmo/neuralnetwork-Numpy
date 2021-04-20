#!/usr/bin/env python3
import numpy as np
from NeuralNetwork import NeuralNetwork

if __name__ == "__main__":

    # create a dataset to train a network for the sum operation
    inputs = np.random.rand(1000,2) * 0.5
    targets = inputs[:,0] + inputs[:,1]

    # create a Multilayer Perceptron with one hidden layer
    network = NeuralNetwork(num_inputs=2, hidden_layers=[5], num_outputs=1)
    # first_input = inputs[0]
    # first_output = targets[0]
    # output = network.feed_forward(inputs=first_input)
    # network.back_propagate(2*(output - first_output))
    # loss_function1 = network.loss_function(target=first_output, output=output)
    # network.weights[1][2,0] += 1e-10
    # output2 = network.feed_forward(inputs=first_input)
    # loss_function2 = network.loss_function(target=first_output, output=output2)
    # derivative = (loss_function2 - loss_function1) / 1e-10
    # print(derivative)
    # print(network.weight_derivatives[1][2,0])

    #exit()
    # train network
    network.train(inputs=inputs, targets=targets, epochs=100, learning_rate=0.05)

    # create dummy data
    input = np.array([0.3, 0.1])
    target = np.array([0.4])

    # get a prediction
    output = network.feed_forward(inputs=input)

    print()
    print("Our network believes that {} + {} is equal to {}".format(input[0], input[1], output[0]))


