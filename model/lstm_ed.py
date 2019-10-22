from sklearn import preprocessing
from utils.utils import binary_matrix
import random
import numpy as np
from numpy import array
from utils.constant import LOAD_AREAS

def prepare_data(data, l=24, r=0.8, h=1, p=0.6):
    bm = binary_matrix(r, len(LOAD_AREAS), data.shape[0])
    X, Y = list(), list()

    for load_area in LOAD_AREAS:
        data_load_area = data[[load_area]]
        T = int(len(data_load_area) * p)
        data_load_area = data_load_area[0:T]

        # x' - standard deviation for the training set
        x_stdev = np.std(data_load_area)
        for i in range (T - l - h):
            x_input = data_load_area[i:i+l].to_numpy()
            y_input = data_load_area[i+l:i+l+h].to_numpy()
            for row in range(len(x_input)):
                if bm[row+i][LOAD_AREAS.index(load_area)] == 1:
                    tmp = x_input[row]
                    x_input[row] = random.uniform((tmp - x_stdev), (tmp + x_stdev))

        X.append(x_input)
        Y.append(y_input)
    return X, Y