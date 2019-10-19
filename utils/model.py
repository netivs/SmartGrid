from sklearn.metrics import mean_squared_error
import numpy as np
import random

def cal_error(test_arr, prediction_arr):
    # cal mse
    error_mse = mean_squared_error(test_arr, prediction_arr)
    print('Test MSE: %.3f' % error_mse)

    # cal rmse
    n = len(prediction_arr)
    error_rmse = np.linalg.norm(prediction_arr - test_arr) / np.sqrt(n)
    print('Test RMSE: %.3f' % error_rmse)

    # cal mape
    y_true, y_pred  = np.array(test_arr), np.array(prediction_arr)
    error_mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print('Test MAPE: %.3f' % error_mape)

def binary_matrix(r, row, col):
    arr = np.zeros((row, col))
    total = 1
    denominator = row * col
    while (total/denominator <= r):
        rnd_row = random.randint(0, row-1)
        rnd_col = random.randint(0, col-1)
        arr[rnd_row][rnd_col] = 1
        total += 1
    return arr