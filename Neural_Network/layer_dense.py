#layer_dense.py
from layer import Layer
import numpy as np

class Layer_Dense(Layer):
    def __init__(self, n_inputs, n_neurons):
        #randn : 0으로 중심으로한 가우스 분포를 무작위 생성
        #실제 randn 함수는 1을 초과한 값을 출력하기 때문에 0.10을 곱함
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        #뉴런 수만 큼 무작위로 편향(값) 생성
        self.biases = np.random.rand(1,n_neurons)
        self.type = 'layer'

    def forward(self, inputs):
        self.input = inputs
        #순방향 전파 수식을 표현
        self.output = np.dot(inputs, self.weights) + self.biases

        return self.output

    def backward(self, d_activation, learning_rate):
        #learning_rate = 0.15
        #delta
        d_input = np.dot(d_activation,self.weights.T)
        #가중치 오류률 산출
        d_weight = np.dot(self.input.T,d_activation)
        #편향 오류률 산출
        d_bias = d_activation.mean(axis=0,keepdims=True)

        #가중치, 편향 업데이트
        self.weights -= learning_rate * d_weight
        self.biases -= learning_rate * d_bias

        return d_input
