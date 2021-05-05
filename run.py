#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from neuralnetwork import NeuralNetwork
from plot import NetworkPlot, LossPlot
from activation_function import HyperbolicTangent


def predict_addition_subtraction_of_two_numbers(network, test_input):
    output = network.predict(inputs=test_input)
    target = [test_input[0] + test_input[1], test_input[0] - test_input[1]]
    print()
    print(" Network: {} + {} = {:+.5f} ({:+.2f})".format(
        test_input[0], test_input[1], output[0][0], target[0])
    )
    print(" Network: {} - {} = {:+.5f} ({:+.2f})".format(
        test_input[0], test_input[1], output[0][1], target[1])
    )
    print(" Network: Loss function = {:.7f}".format(
        network.loss_function(target=target, output=output))
    )


if __name__ == "__main__":

    # create a Multilayer Perceptron with one hidden layer
    network = NeuralNetwork(num_inputs=2, hidden_layers=[4, 5], num_outputs=2,
                            activation_function=HyperbolicTangent())

    # create a dataset to train a network for the sum operation
    inputs = np.random.rand(500, 2) * 0.5
    targets = np.stack((inputs[:, 0] + inputs[:, 1], inputs[:, 0] - inputs[:, 1]), axis=1)

    # train network
    num_iterations = 100
    # Create a generator for training
    trainer = network.train(inputs=inputs, targets=targets, epochs=num_iterations,
                            learning_rate=0.01, batch_size=1)
    # test data for live plot
    test_inputs = np.random.rand(100, 2) * 0.5
    test_targets = np.stack((test_inputs[:, 0] + test_inputs[:, 1],
                             test_inputs[:, 0] - test_inputs[:, 1]), axis=1)

    # Train the network and plot the results
    fig = plt.figure(figsize=(14, 6))
    plot1 = LossPlot(fig)
    plot2 = NetworkPlot(fig)
    pause = 0.0001
    # For the network
    training_loss = []
    test_loss = []
    for loss in tqdm(trainer, total=num_iterations, desc="Training neural network:"):
        test_outputs = network.predict(test_inputs)
        test_loss.append(network.loss_function(test_targets, test_outputs))
        training_loss.append(loss)
        plot1.plot(training_loss, test_loss, pause)
        plot2.plot(layers=network.layers, weights=network.weights, biases=network.biases, pause=pause)

    print()
    print(" Now let's do some tests... :-)")
    predict_addition_subtraction_of_two_numbers(network, [0.30, 0.10])
    predict_addition_subtraction_of_two_numbers(network, [0.24, 0.36])
    predict_addition_subtraction_of_two_numbers(network, [0.11, 0.42])
    predict_addition_subtraction_of_two_numbers(network, [0.27, 0.27])

    plt.show()
