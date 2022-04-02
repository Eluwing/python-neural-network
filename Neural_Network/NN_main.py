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

# Generate Model Object
model = Model()
model.add(Layer_Dense(2,4))
model.add(ReLU(4))
model.add(Layer_Dense(4,3))
model.add(Softmax(3))

#Set loss function with Categorical Cross Entropy
model.setLoss(loss_function)

#Train model object
model.train(X, y_true, iteration = 1000, learning_rate = 0.15)

#Test
out = model.predict_loss(X, y_true)
print(out)
