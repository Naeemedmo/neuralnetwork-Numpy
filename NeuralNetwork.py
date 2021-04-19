
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def d_sigmoid(sigmoid_x):
    '''
    sigmoid_x is the same as sigmoid(x)
    '''
    return sigmoid_x * (1.0 - sigmoid_x)


class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.n_nodes_hiddenlayer = 4
        self.n_output = 1
        self.weights1 = np.random.rand(self.input.shape[1], self.n_nodes_hiddenlayer)
        self.biases1 = np.random.rand(1, self.n_nodes_hiddenlayer)
        self.weights2 = np.random.rand(self.n_nodes_hiddenlayer, self.n_output)
        self.biases2 = np.random.rand(1, self.n_output)
        self.y = y
        self.output = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1) + self.biases1)
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2) + self.biases2)
        return self.layer2

    def backprop(self):
        '''
        Loss(y_output - y ) = sigma(y_output- y)^2
        apply chain rule to get derivitive in respect to w2,w1,b2,b1
        self.output = r = w2.z + b2
        self.layer1 = q = w1.x + b1
        z = sigmoid(r)
        y = sigmoid(q)
        '''
        #initializing derivitives...
        self.d_weights1 = np.zeros((self.input.shape[1], self.n_nodes_hiddenlayer))
        self.d_biases1 = np.zeros((1, self.n_nodes_hiddenlayer))
        self.d_weights2 = np.zeros((self.n_nodes_hiddenlayer, self.n_output))
        self.d_biases2 = np.zeros((1, self.n_output))
        for n in range(self.input.shape[0]):
            #chain rule components
            for node in range(self.n_output):
                d_loss_y = 2.0 * (self.y[n,node] - self.layer2[n,node])
                self.d_biases2[0,node] = d_loss_y * d_sigmoid(self.layer2[0,node]) * 1.0
                for previous_node in range(self.n_nodes_hiddenlayer):
                    self.d_weights2[previous_node,node] = self.d_biases2[0,node] * self.layer1[0,node]

            for node in range(self.n_nodes_hiddenlayer):
                for next_node in range(self.n_output):
                    self.d_biases1[0,node] += self.d_biases2[0,next_node] * self.weights2[node,next_node] * d_sigmoid(self.layer1[0,node]) * 1.0
                    for previous_node in range(self.input.shape[1]):
                        self.d_weights1[previous_node,node] += self.d_biases1[0,node] * self.input[0,previous_node]

    #update the weights and biases with derivative of loss function
    def update(self,learning_rate=0.01):
        self.weights1 += self.d_weights1 * learning_rate
        self.weights2 += self.d_weights2 * learning_rate
        self.biases1 += self.d_biases1 * learning_rate
        self.biases2 += self.d_biases2 * learning_rate

    def train(self, x, y):
        self.output = self.feedforward()
        self.backprop()
        self.update()

