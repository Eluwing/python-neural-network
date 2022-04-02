#losses.py
import numpy as np

class Loss_mse:
    def forward(self,y_true,y_pred):
        return np.mean(np.power(y_true-y_pred, 2));

    def backward(self,y_true, y_pred):
        return 2*(y_pred-y_true)/y_true.size;

class Loss_CCE():
    def forward(self, output, y):
        #Number of predicted probability distributions
        n = output.shape[0]
        cce_matrix = self.CCE_forward(output, y)
        loss_cce = cce_matrix/n
        return loss_cce

    def backward(self, output, y):
        return self.CCE_backward(output,y)

class Loss_CategoricalCrossentropy(Loss_CCE):

    def CCE_forward(self, y_pred, y_true):
        #avoid overflow
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        loss_cce = -np.sum(y_true*np.log(y_pred_clipped))

        return loss_cce

    def CCE_backward(self,y_pred, y_true):
        n = y_true.shape[0]

        # When the true probability distribution is not one-HotEncoding
        if len(y_true.shape) == 1:
            gradient = y_pred
            gradient[range(n),y_true] -= 1
            gradient = gradient/n
        # When the true probability distribution is one-HotEncoding
        elif len(y_true.shape) == 2:
            gradient = (y_pred-y_true)/n

        return gradient
