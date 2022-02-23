#tools.py
import numpy as np

class NN_tools:
    def oneHotEncoding(y_true):
        rowLen = len(y_true)
        colLen = max(y_true)+1
        array = [[0 for col in range(colLen)] for row in range(rowLen)]

        i = 0
        for val in y_true:
            array[i][val]= 1
            i +=1

        return np.array(array)
