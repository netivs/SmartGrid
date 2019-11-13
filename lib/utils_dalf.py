import logging
import os
import sys
import csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime

#### D = NUM_HOUR
def load_dataset(seq_len, horizon, num_hour, input_dim, output_dim, raw_dataset_dir, verified_percentage, p, **kwargs):
    raw_data = np.load(raw_dataset_dir)['data']

    print('|--- Splitting train-test set.')
    train_data2d, valid_data2d, test_data2d = prepare_train_valid_test_2d(data=raw_data, p=p)
    print('|--- Normalizing the train set.')
    data = {}
    scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
    scaler.fit(train_data2d)
    train_data2d_norm = scaler.transform(train_data2d)
    valid_data2d_norm = scaler.transform(valid_data2d)
    test_data2d_norm = scaler.transform(test_data2d)

    data['test_data_norm'] = test_data2d_norm.copy()

    encoder_input_train, decoder_input_train, decoder_target_train = create_data(train_data2d_norm,
                                                                                seq_len=seq_len,
                                                                                verified_percentage=verified_percentage,
                                                                                num_hour = num_hour,
                                                                                input_dim=input_dim,
                                                                                output_dim=output_dim,
                                                                                horizon=horizon)
    encoder_input_val, decoder_input_val, decoder_target_val = create_data(valid_data2d_norm,
                                                                                seq_len=seq_len, 
                                                                                verified_percentage=verified_percentage,
                                                                                num_hour=num_hour,
                                                                                input_dim=input_dim,
                                                                                output_dim=output_dim,
                                                                                horizon=horizon)
    encoder_input_eval, decoder_input_eval, decoder_target_eval = create_data(test_data2d_norm,
                                                                                seq_len=seq_len, 
                                                                                verified_percentage=verified_percentage,
                                                                                num_hour=num_hour,
                                                                                input_dim=input_dim,
                                                                                output_dim=output_dim,
                                                                                horizon=horizon)

    for cat in ["train", "val", "eval"]:
        e_x, d_x, d_y = locals()["encoder_input_" + cat], locals()[
            "decoder_input_" + cat], locals()["decoder_target_" + cat]
        print(cat, "e_x: ", e_x.shape, "d_x: ", d_x.shape, "d_y: ", d_y.shape)
        data["encoder_input_" + cat] = e_x
        data["decoder_input_" + cat] = d_x
        data["decoder_target_" + cat] = d_y
    data['scaler'] = scaler

    return data

def prepare_train_valid_test_2d(data, p=0.6):
    train_size = int(data.shape[0] * p)
    valid_size = int(data.shape[0] * 0.2)

    train_set = data[0:train_size]
    valid_set = data[train_size: train_size + valid_size]
    test_set = data[train_size + valid_size:]

    return train_set, valid_set, test_set

def create_data(data, seq_len, verified_percentage, num_hour, input_dim, output_dim, horizon):
    K = data.shape[1]
    T = data.shape[0]
    bm = binary_matrix(verified_percentage, T, K)
    _data = data.copy()
    _std = np.std(data)

    _data[bm == 0] = np.random.uniform(_data[bm == 0] - _std, _data[bm == 0] + _std)

    en_x = np.zeros(shape=((T - seq_len*num_hour - horizon) * K, seq_len, input_dim))
    de_x = np.zeros(shape=((T - seq_len*num_hour - horizon) * K, horizon + 1, output_dim))
    de_y = np.zeros(shape=((T - seq_len*num_hour - horizon) * K, horizon + 1, output_dim))

    for k in range(K):
        for i in range(T - seq_len*num_hour - horizon):
            # _data[start:stop:step]
            en_x[i, :, 0] = _data[i:(i + seq_len*num_hour):num_hour, k]
            en_x[i, :, 1] = bm[i:(i + seq_len*num_hour):num_hour, k]
            # en_x[i, :, 2] = 

            de_x[i, 0, 0] = 0
            de_x[i, 1, 0] = data[i + (seq_len-1)*num_hour, k]
            de_y[i, 0, 0] = data[i + (seq_len-1)*num_hour, k]
            de_y[i, 1, 0] = data[i + seq_len*num_hour, k]
    return en_x, de_x, de_y


def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger

def cal_error(test_arr, prediction_arr):
    with np.errstate(divide='ignore', invalid='ignore'):
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
        error_list = [error_mae, error_rmse, error_mape]
        return error_list


def binary_matrix(r, row, col):
    tf = np.array([1, 0])
    bm = np.random.choice(tf, size=(row, col), p=[r, 1.0 - r])
    return bm


def save_metrics(error_list, log_dir, alg):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    error_list.insert(0, dt_string)
    with open(log_dir + alg + "_metrics.csv", 'a') as file:
        writer = csv.writer(file)
        writer.writerow(error_list)