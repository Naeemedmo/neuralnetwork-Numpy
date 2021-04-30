
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
