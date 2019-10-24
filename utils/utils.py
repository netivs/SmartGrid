import logging
import numpy as np
import os
import random
import pandas as pd
import pickle
import sys
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.constant import LOAD_AREAS

class StandardScaler:
    """
    Standard the input
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


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


def prepare_train_valid_test_2d(data, p=0.6):
    # len(data_load_area) = data.shape[0]
    train_size = int(data.shape[0] * p)
    valid_size = int(data.shape[0] * 0.2)

    train_set = data[0:train_size]
    valid_set = data[train_size: train_size+valid_size]
    test_set = data[train_size+valid_size:]

    return train_set, valid_set, test_set


def create_data_lstm_ed(data, seq_len, r, input_dim=1, horizon=1):
    K = data.shape[1]
    T = data.shape[0]
    bm = binary_matrix(r, K, T)
    e_x, d_x, d_y = list(), list(), list()
    for col in range(K):
        data_load_area = data[:, col]
        # x' - standard deviation for the training set
        x_stdev = np.std(data_load_area)
        for i in range (T - seq_len - horizon):
            x_en = data_load_area[i:i+seq_len]
            x_de = data_load_area[i+seq_len-1:i+seq_len+horizon-1]
            y_de = data_load_area[i+seq_len:i+seq_len+horizon]
            for row in range(len(x_en)):
                if bm[row+i][col] == 1:
                    tmp = x_en[row]
                    x_en[row] = random.uniform((tmp - x_stdev), (tmp + x_stdev))

            for row in range(len(x_de)):
                if bm[row+i+seq_len-1][col] == 1:
                    tmp = x_de[row]
                    x_de[row] = random.uniform((tmp - x_stdev), (tmp + x_stdev))

            for row in range(len(y_de)):
                if bm[row+i+seq_len][col] == 1:
                    tmp = y_de[row]
                    y_de[row] = random.uniform((tmp - x_stdev), (tmp + x_stdev))

            x_en = x_en.reshape(seq_len, input_dim)
            x_de = x_de.reshape(horizon, input_dim)
            y_de = y_de.reshape(horizon, input_dim)
            e_x.append(x_en)
            d_x.append(x_de)
            d_y.append(y_de)
    e_x = np.stack(e_x, axis=0)
    d_x = np.stack(d_x, axis=0)
    d_y = np.stack(d_y, axis=0)
    return e_x, d_x, d_y

def load_dataset_lstm_ed(seq_len, horizon, input_dim, raw_dataset_dir, r, p, **kwargs):
    raw_data = np.load(raw_dataset_dir)['data']

    print('|--- Splitting train-test set.')
    train_data2d, valid_data2d, test_data2d = prepare_train_valid_test_2d(data=raw_data, p=p)
    print('|--- Normalizing the train set.')
    data = {}
    scaler = StandardScaler(mean=train_data2d.mean(), std=train_data2d.std())
    train_data2d_norm = scaler.transform(train_data2d)
    valid_data2d_norm = scaler.transform(valid_data2d)
    test_data2d_norm = scaler.transform(test_data2d)

    data['test_data_norm'] = test_data2d_norm

    encoder_input_train, decoder_input_train, decoder_target_train = create_data_lstm_ed(train_data2d_norm,
                                                seq_len=seq_len, r=r, input_dim=input_dim, horizon=horizon)
    encoder_input_val, decoder_input_val, decoder_target_val = create_data_lstm_ed(valid_data2d_norm,
                                                seq_len=seq_len, r=r, input_dim=input_dim, horizon=horizon)
    encoder_input_eval, decoder_input_eval, decoder_target_eval = create_data_lstm_ed(test_data2d_norm,
                                                seq_len=seq_len, r=r, input_dim=input_dim, horizon=horizon)

    for cat in ["train", "val", "eval"]:
        e_x, d_x, d_y = locals()["encoder_input_" + cat], locals()[
            "decoder_input_" + cat], locals()["decoder_target_" + cat]

        data["encoder_input_" + cat] = e_x
        data["decoder_input_" + cat] = d_x
        data["decoder_target_" + cat] = d_y
    data['scaler'] = scaler

    return data


def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

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