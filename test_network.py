#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork

if __name__ == "__main__":

    #Inputs
    #-----------------------------------------------------------------------
    # Each row is a training example, each column is a feature  [X1, X2, X3]
    #x_input = np.array(([0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]), dtype=float)
    #y_output = np.array(([0], [1], [1], [0]), dtype=float)
    x_input = np.array([0, 0, 1], dtype=float).reshape(1,3)
    y_output = np.array([0], dtype=float).reshape(1,1)
    n_iterations = 1500

    #test network
    my_network = NeuralNetwork(x_input, y_output)
    errors = []

    for i in range(n_iterations):
        predicted_output = my_network.feedforward()
        errors.append(np.mean(np.square(y_output - predicted_output)))
        if i % 100 == 0:
           	print ("Iteration {}".format(i))
           	print ("Input: {}".format(x_input))
           	print ("Actual Output {}".format(y_output))
           	print ("Predicted Output: {}".format(predicted_output))
           	print ("Loss: {}".format(errors[i])) # mean sum squared loss
        my_network.train(x_input, y_output)

    #Plot the errors
    plt.plot(errors)
    plt.show()
