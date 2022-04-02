#layer_dense.py
from layer import Layer
import numpy as np

class Layer_Dense(Layer):
    def __init__(self, n_inputs, n_neurons):
        #randn : Randomly generate a zero-centered Gaussian distribution
        #randn function multiplies by 0.10 because it outputs a value greater than 1.
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        #Random generate bias values as many as the number of neurons
        self.biases = np.random.rand(1,n_neurons)
        self.type = 'layer'

    def forward(self, inputs):
        self.input = inputs
        #Forward propagation formula
        self.output = np.dot(inputs, self.weights) + self.biases

        return self.output

    def backward(self, d_activation, learning_rate):

        #delta
        d_input = np.dot(d_activation,self.weights.T)

        #Calculate the error rate of weight
        d_weight = np.dot(self.input.T,d_activation)

        #Calculate the error rate of bias
        d_bias = d_activation.mean(axis=0,keepdims=True)

        #Update to weights and biases of layer
        self.weights -= learning_rate * d_weight
        self.biases -= learning_rate * d_bias

        return d_input
