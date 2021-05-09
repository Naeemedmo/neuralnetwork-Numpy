import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


class LossPlot():
    '''
        Left subplot: plot averaged loss function over iterations
    '''
    def __init__(self, fig):
        self.ax = fig.add_subplot(1, 2, 1)

    def plot(self, data, test_data, pause):
        self.ax.clear()
        self.ax.set_title('Mean loss function over iterations')
        self.ax.set_xlabel('Iterations')
        self.ax.set_ylabel('Loss function')
        self.ax.set_yscale('log')
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax.plot(data, label='training error')
        self.ax.plot(test_data, label='testing error')
        self.ax.legend()
        plt.pause(pause)


class NetworkPlot():
    '''
        Right subplot: draw the network neurons and connections.
        The linewidth of line represent the effect of weights for each iteration
        The marker on the nodes represent the effect of biases for each iteration
        For both blueish color is + and the other ones negative
    '''
    # fixed size for figure
    h_max = 12.0
    v_max = 12.0
    side_space = 1.0
    top_space = 1.0
    # radius for the nodes
    radius = 0.35
    # line thickness of weights
    max_linewidth = 3.0
    min_linewidth = 0.5
    # markersize for biases (area)
    max_markersize = 4
    min_markersize = 2

    def __init__(self, fig):
        self.ax = fig.add_subplot(1, 2, 2)
        # initialize the min/max
        self.max_weights = -float("inf")
        self.min_weights = float("inf")
        self.max_biases = -float("inf")
        self.min_biases = float("inf")

    def iteration_bounds(self, weights, biases):
        max_weights = np.max([np.max(np.abs(w)) for w in weights])
        min_weights = np.min([np.min(np.abs(w)) for w in weights])
        max_biases = np.max([np.max(np.abs(b)) for b in biases])
        min_biases = np.min([np.min(np.abs(b)) for b in biases])
        return min_weights, max_weights, min_biases, max_biases

    def update_bounds(self, weights, biases):
        min_weights, max_weights, min_biases, max_biases = self.iteration_bounds(weights, biases)
        self.max_weights = max(self.max_weights, max_weights)
        self.min_weights = min(self.min_weights, min_weights)
        self.max_biases = max(self.max_biases, max_biases)
        self.min_biases = min(self.min_biases, min_biases)

    def draw_nodes_of_layer(self, i_layer, num_nodes, radius):
        for n in range(num_nodes):
            circle = plt.Circle((self.h_levels[i_layer], self.v_levels[i_layer][n]),
                                radius=radius, color='#F5DD90', zorder=100)
            self.ax.add_patch(circle)

    def draw_weights_of_layer_leftlayer(self, i_layer, weights):
        for n in range(self.layers[i_layer]):
            for nn in range(self.layers[i_layer-1]):
                weight = abs(weights[i_layer-1][nn][n])
                linewidth = self.m_linewidth_w * (weight - self.min_weights) + self.min_linewidth
                x_values = [self.h_levels[i_layer-1], self.h_levels[i_layer]]
                y_values = [self.v_levels[i_layer-1][nn], self.v_levels[i_layer][n]]
                line_color = '#586BA4' if weights[i_layer-1][nn][n] > 0.0 else '#F76C5E'
                plt.plot(x_values, y_values, zorder=1, color=line_color, linewidth=linewidth)

    def draw_biases_of_layer(self, i_layer, num_nodes, biases):
        for n in range(num_nodes):
            marker_size = self.m_marker_b * (biases[i_layer][n] - self.min_biases) + self.min_markersize
            marker_color = '#324376' if biases[i_layer][n] > 0.0 else '#F68E5F'
            plt.plot(
                self.h_levels[i_layer+1],
                self.v_levels[i_layer+1][n],
                zorder=200,
                marker="s",
                color=marker_color,
                markersize=marker_size**2
            )

    def plot(self, layers, weights, biases, pause):

        # update the min, max of weights and biases
        self.update_bounds(weights, biases)
        # linear interpolation forlinewidth, and markersize y - y0 = m (x -x0)
        self.m_linewidth_w = (self.max_linewidth - self.min_linewidth) / (self.max_weights - self.min_weights)
        self.m_marker_b = (self.max_markersize - self.min_markersize) / (self.max_biases - self.min_biases)
        self.layers = layers
        # clear the plot, set axes
        self.ax.clear()
        self.ax.set_xlim((0.0, self.h_max))
        self.ax.set_ylim((0.0, self.v_max))

        # get information on layers for figure
        num_nodes_height = max(self.layers)  # layer with the most nodes define height
        num_nodes_width = len(self.layers)  # number of layers define width

        h_center = self.h_max / 2.0
        v_center = self.v_max / 2.0
        # spacing between nodes
        h_space = (self.h_max - 2 * self.side_space) / (num_nodes_width - 1)
        v_space = (self.v_max - 2 * self.top_space) / (num_nodes_height - 1)
        # layers horizontal position
        h_start = h_center - ((num_nodes_width - 1) / 2) * h_space
        self.h_levels = [h_start + n * h_space for n in range(len(self.layers))]
        # vertical position of each node
        self.v_levels = [np.zeros((layer)) for layer in self.layers]
        for layer, num_nodes in enumerate(self.layers):
            # calculate position of the toppest node
            v_start = v_center + ((num_nodes - 1) / 2) * v_space
            self.v_levels[layer] = [v_start - n * v_space for n in range(num_nodes)]

        # first make the input nodes
        self.draw_nodes_of_layer(i_layer=0, num_nodes=self.layers[0], radius=self.radius)
        for layer, num_nodes in enumerate(self.layers[1:]):
            # first markers for biases
            self.draw_biases_of_layer(layer, self.layers[layer+1], biases)
            layer += 1
            self.draw_nodes_of_layer(i_layer=layer, num_nodes=self.layers[layer], radius=self.radius)
            # connect the node to the previous layer
            self.draw_weights_of_layer_leftlayer(i_layer=layer, weights=weights)
        plt.axis('off')
        # print information on the corner
        min_weights, max_weights, min_biases, max_biases = self.iteration_bounds(weights, biases)
        information = f'max(abs(weights)) = {max_weights:.5f} \n'
        information += f'min(abs(weights)) = {min_weights:.5f} \n'
        information += f'max(abs(biases)) = {max_biases:.5f} \n'
        information += f'min(abs(biases)) = {min_biases:.5f} \n'
        self.ax.text(
            x=1.2,
            y=-0.1,
            s=information,
            verticalalignment='bottom',
            horizontalalignment='right',
            transform=self.ax.transAxes,
            color='k',
            fontsize=8,
            fontstyle='oblique'
            )
        plt.pause(pause)
