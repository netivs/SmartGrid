import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def cal_error_all_load_areas(metrics_all_load_area):
    for load_area in metrics_all_load_area:
        print("TEST METRIC FOR ", load_area[0])
        cal_error(load_area[1], load_area[2])
        print("\n\n")


def cal_error(test_arr, prediction_arr):
    # for test, prediction in range(test_arr, prediction_arr)
    # cal mse
    error_mae = mean_absolute_error(test_arr, prediction_arr)
    print('MAE: %.3f' % error_mae)

    # cal rmse
    error_mse = mean_squared_error(test_arr, prediction_arr)
    error_rmse = np.sqrt(error_mse)
    print('RMSE: %.3f' % error_rmse)

    # cal mape
    y_true, y_pred = np.array(test_arr), np.array(prediction_arr)
    error_mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print('MAPE: %.3f' % error_mape)


def binary_matrix(r, row, col):
    tf = np.array([1, 0])
    bm = np.random.choice(tf, size=(col, row), p=[r, 1.0 - r])
    return bm
