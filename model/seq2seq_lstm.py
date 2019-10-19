from sklearn import preprocessing
from utils.model import binary_matrix
import random
import numpy as np
from numpy import array

def prepare_data(train, l, r = 0.8, h = 1, p = 0.01):
    # get specific numbers of time-series
    X, Y = list(), list()
    train = train.transpose()
    T = int(train.shape[1] * p)
    train = train.iloc[:, 0:T]

    rows = train.shape[0]
    cols = train.shape[1]
    
    # Data normalization
    train = preprocessing.normalize(train)
    for in_start in range(0, len(train) - l - h + 1):        
        # Data need to be transformed as format: (x,y) where x is the input with shape (K,l), y is the targetdasd with shape (K,h)
        in_end = in_start + l
        out_start = in_end
        out_end = in_end + h
        x_input = train[:, in_start:in_end]
        y_input = train[:, out_start:out_end]
        print(x_input.shape)
        print(y_input.shape)
        
        # Randomly create the binary matrix M_(K×l) which (∑(M) )/(K×l)=r
        binary_mat = binary_matrix(r, rows, cols)
        # Change the value x_i^k whose m_i^k=0 as x_i^k←random(x_i^k-x ´,x_i^k+x ´ ), where x ´ is the stdev of the training set
        for k in range(0, rows):
            # standard deviation for kth training set
            x_stdev = np.std(train[k])
            for i in range(in_start, in_end):
                if binary_mat[k][i] == 1:
                    tmp = x_input[k][i-in_start]
                    x_input[k][i-in_start] = random.uniform((tmp - x_stdev), (tmp + x_stdev))
        X.append(x_input)
        Y.append(y_input)
    # Add to (x, y) to data_loader

    return array(X), array(Y)