import numpy as np


class ActivationFunction():
    '''
    The neural network needs the function and its derivative in this style
    '''
    def evaluate(self, x):
        raise Exception("not implemented!")

    def derivative(self, f_x):
        raise Exception("not implemented!")


class Sigmoid(ActivationFunction):

    def evaluate(self, x):
        '''
        The function takes any real value as input and outputs values in the range 0 to 1.
        '''
        return 1.0 / (1.0 + np.exp(-x))

    def derivative(self, sigmoid_x):
        '''
        sigmoid_x is the same as sigmoid(x)
        '''
        return sigmoid_x * (1.0 - sigmoid_x)


class HyperbolicTangent(ActivationFunction):

    def evaluate(self, x):
        '''
        The function takes any real value as input and outputs values in the range -1 to 1.
        '''
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def derivative(self, tanh_x):
        '''
        tanh_x is the same as tanh(x)
        '''
        return 1 - tanh_x ** 2
