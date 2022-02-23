#activation_layer.py
import numpy as np
from layer import Layer

# activation function and its derivative

class Tanh:
    def __init__(self):
        self.forward_value = None

    def forward(self,x):
        self.forward_value = np.tanh(x)
        return self.forward_value

    def backward(self,x):
        forward_tanh =  self.forward_value
        return 1-forward_tanh**2;

class Sigmoid:
    def __init__(self):
        self.forward_value = None

    def forward(self,x):
        self.forward_value = (1/(1 + np.exp(-inputs)))
        return self.forward_value

    def backward(self,x):
        forward_sig = self.forward_value
        self.output_back = gradinet * forward_sig * (1 - forward_sig)

class Softmax(Layer):
    def __init__(self, output_cnt):

        self.units = output_cnt
        self.type = 'Softmax'

    def forward(self,inputs):
        #오버플로우를 방지하기 위해 최대값을 빼줌
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

        return self.output

    def backward(self,gradient):
        forward_soft = self.output
        self.output_back = forward_soft * (gradient -( gradient * forward_soft).sum(axis=1, keepdims=True))

        return self.output_back

class ReLU(Layer):
    def __init__(self, output_cnt):

        self.units = output_cnt
        self.type = 'ReLU'

    def forward(self,inputs):
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self,gradient):
        foward_ReLU = self.output

        self.output_back = gradient * np.where(foward_ReLU<=0, 0, 1)
        #self.output_back = gradient * np.heaviside(foward_ReLU,0)

        return self.output_back
