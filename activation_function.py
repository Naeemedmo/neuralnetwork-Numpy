import numpy as np


class ActivationFunction():
    """
    The neural network needs the function and its derivative in this style
    """
    def evaluate(self, x: float) -> float:
        """ Activation function

        Args:
            x: input value

        Returns:
            :math:`f(x)`

        """
        raise Exception("not implemented!")

    def derivative(self, f_x: float) -> float:
        """ Derivative of the Activation function (:math:`\\frac{df}{dx}`)

        Args:
            f_x: :math:`f(x)` returned by :meth:`evaluate`

        Returns:
            :math:`f'(x)`

        """
        raise Exception("not implemented!")


class Sigmoid(ActivationFunction):

    def evaluate(self, x: float) -> float:
        """ Sigmoid function.
        It takes any real value as input and outputs values in the range 0 to 1.

        Args:
            x: input value

        Returns:
            :math:`S(x) = \\frac{1}{1+e^{-x}}`

        """
        return 1.0 / (1.0 + np.exp(-x))

    def derivative(self, sigmoid_x: float) -> float:
        """ Derivative of Sigmoid function (:math:`\\frac{dS}{dx}`)

        Args:
            sigmoid_x::math:`S(x)` returned by :meth:`evaluate`

        Returns:
            :math:`S'(x) = S(x)(1-S(x))`

        """
        return sigmoid_x * (1.0 - sigmoid_x)


class HyperbolicTangent(ActivationFunction):

    def evaluate(self, x: float) -> float:
        """ Hyperbolic Tangent function.
        It takes any real value as input and outputs values in the range -1 to 1.

        Args:
            x: input value

        Returns:
            :math:`tanh(x) = \\frac{e^{x} - e^{-x}}{e^{x}+e^{-x}}`

        """
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def derivative(self, tanh_x: float) -> float:
        """ Derivative of Hyperbolic Tangent function (:math:`\\frac{dtanh}{dx}`)

        Args:
            tanh_x: :math:`tanh(x)` returned by :meth:`evaluate`

        Returns:
            :math:`tanh'(x) = 1-tanh(x)^2`

        """
        return 1 - tanh_x ** 2
