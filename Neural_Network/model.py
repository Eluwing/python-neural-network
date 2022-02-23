#model.py
class Model:
    def __init__(self):
        self.layers = []
        self.loss_value = []
        self.loss = None

    def add(self, layer):
        self.layers.append(layer)

    def setLoss(self, loss_function):
        self.loss = loss_function

    def predict(self, n_inputs):
        for i, _ in enumerate(self.layers):
            layer_forward = self.layers[i].forward(n_inputs)
            n_inputs = layer_forward

        return layer_forward

    def predict_loss(self, n_inputs, y_true):

        predict_result = self.predict(n_inputs)
        loss = self.loss.forward(predict_result, y_true)

        return loss

    def train(self, x_train, y_train, iteration, learning_rate):

        for i in range(iteration):

            #순방향 전파
            layer_forward = self.predict(x_train)

            #순방향 값의 손실 값 계산
            loss = self.loss.forward(layer_forward, y_train)

            self.loss_value.append(loss)

            gradient = self.loss.backward(layer_forward, y_train)

            #역전파
            for z, _ in reversed(list(enumerate(self.layers))):
                #print("z:",z)
                if self.layers[z].type == 'layer':
                    gradient = self.layers[z].backward(gradient,learning_rate)
                else:
                    gradient = self.layers[z].backward(gradient)

            print('iteration %d/%d   loss=%f' % (i+1, iteration, loss))
