import csv
import numpy as np
import random
from matplotlib import pyplot as plt


class Neuron:
    def __init__(self, num_weights: int = 1, bias: int = 0):
        self.weights = np.array([1 for _ in range(num_weights)])
        self.bias = bias

    def __str__(self):
        return f"bias: {self.bias} | weights: {self.weights}"

    def __repr__(self):
        return f"bias: {self.bias} | weights: {self.weights}"


class PyNet:
    def __init__(self, input_size: int = 1, hidden_size: int = 1, num_hidden: int = 1, output_size: int = 1,
                activation_func: str = None):
        if activation_func is None:
            activation_func = "relu"

        # first hidden layer
        layers = [[Neuron(num_weights=input_size) for _ in range(hidden_size)]]
        # other hidden layers
        for i in range(1, num_hidden):
            new_layer = [Neuron(num_weights=hidden_size) for _ in range(hidden_size)]
            layers.append(new_layer)
        # output layer
        layers.append([Neuron(num_weights=hidden_size) for _ in range(output_size)])

        self.input_size = input_size
        self.output_size = output_size
        self.activation_func = activation_func
        self.layers = layers

    def load_nnet_from_csv(self, path):
        with open(path, newline='') as f:
            reader = csv.reader(f)
        # TODO: add NN state from file functionality

    def test_nnet(self, test_data_path: str):
        input_arr = None
        with open(test_data_path, newline='') as f:
            reader = csv.reader(f)
            headers = next(reader)
            row1 = [int(x) for x in next(reader)]
            label = row1[0]
            img1 = input_arr = np.array(row1[1:])
            img1 = np.reshape(img1, (28, 28))
            plt.imshow(img1, interpolation='nearest')
            plt.show()

        print(self.compute_result(input_arr))

    def compute_result(self, input_layer: list):

        for _, layer in enumerate(self.layers):
            layer_output = []
            for i, neuron in enumerate(layer):
                dot_prod = np.dot(input_layer, neuron.weights)
                # TODO add activation function here
                layer_output.append(dot_prod + neuron.bias)
            input_layer = layer_output

        return input_layer


# need to add activation function for each layer (other list)
# need to add randomized values for layers
# add training
# profit

myNN = PyNet(784, 30, 1, 10)
myNN.test_nnet('dataset/mnist_train.csv')
