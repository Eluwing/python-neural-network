#NN_main.py
import numpy as np

from model import Model
from layer_dense import *
from activation_layer import *
from losses import *
from tools import NN_tools

import nnfs
from nnfs.datasets import spiral_data
from nnfs.datasets import vertical_data

#X, y = spiral_data(samples=100, classes=3)
X, y = vertical_data(samples=100, classes=3)

y_true = NN_tools.oneHotEncoding(y)

loss_function = Loss_CategoricalCrossentropy()

# model 객체 생성
model = Model()
model.add(Layer_Dense(2,4))
model.add(ReLU(4))
model.add(Layer_Dense(4,3))
model.add(Softmax(3))

#범주형 교차 엔트로피로 손실 함수 설정
model.setLoss(loss_function)

#모델 훈련
model.train(X, y_true, iteration = 1000, learning_rate = 0.15)

#테스트
out = model.predict_loss(X, y_true)
print(out)
