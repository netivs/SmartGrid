from sklearn import preprocessing
from utils.utils import binary_matrix
import random
import numpy as np
from numpy import array
from utils.constant import LOAD_AREAS

def prepare_data(data, l=24, r=0.8, h=1, p=0.6):
    row_data = data.shape[0]
    bm = binary_matrix(r, row_data, len(LOAD_AREAS))
    X, Y = list(), list()
    K = len(LOAD_AREAS)
    T = int(row_data * p)
    data = data[0:T]

    for i in range (T - l - h):
        x_input = data.iloc[i:i+l, :].to_numpy()
        y_input = data.iloc[i+l:i+l+h, :].to_numpy()
        for load_area in LOAD_AREAS:
            # x' - standard deviation for the training set
            col = LOAD_AREAS.index(load_area)
            x_stdev = np.std(data[[load_area]])
            for row in range(len(x_input)):
                if bm[row+i][col] == 1:
                    tmp = x_input[row][col]
                    x_input[row][col] = random.uniform((tmp - x_stdev), (tmp + x_stdev))

        x_input = x_input.reshape(l, K, 1)
        y_input = y_input.reshape(h, K, 1)
        X.append(x_input)
        Y.append(y_input)
    
    X = np.stack(X, axis=0)
    Y = np.stack(Y, axis=0)
    return X, Y