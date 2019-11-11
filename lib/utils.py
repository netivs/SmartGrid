import logging
import os
import csv
import random
import pandas as pd
import pickle
import sys
import numpy as np
import tensorflow as tf
import scipy.sparse as sp

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse import linalg
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)

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
    valid_set = data[train_size: train_size + valid_size]
    test_set = data[train_size + valid_size:]

    return train_set, valid_set, test_set


def new_expand_dims(a, axis):
    # if int is passed, retain the same behaviour
    if type(axis) == int:
        return np.expand_dims(a, axis)
    # insert axis to given indices
    for ax in sorted(axis):
        a = np.expand_dims(a, ax)
    return a


def create_data_lstm_ed_ver_cuc_xin(data, seq_len, r, input_dim, output_dim, horizon):
    K = data.shape[1]
    T = data.shape[0]
    bm = binary_matrix(r, T, K)
    _data = data.copy()
    _std = np.std(data)

    _data[bm == 0] = np.random.uniform(_data[bm == 0] - _std, _data[bm == 0] + _std)

    en_x = np.zeros(shape=((T - seq_len - horizon) * K, seq_len, input_dim))
    de_x = np.zeros(shape=((T - seq_len - horizon) * K, horizon + 1, output_dim))
    de_y = np.zeros(shape=((T - seq_len - horizon) * K, horizon + 1, output_dim))

    _idx = 0
    for k in range(K):
        for i in range(T - seq_len - horizon):
            en_x[_idx, :, 0] = _data[i:i + seq_len, k]
            en_x[_idx, :, 1] = bm[i:i + seq_len, k]

            de_x[_idx, 0, 0] = 0
            de_x[_idx, 1:, 0] = data[i + seq_len - 1:i + seq_len + horizon - 1, k]
            de_y[_idx, :, 0] = data[i + seq_len - 1:i + seq_len + horizon, k]

            _idx += 1
    return en_x, de_x, de_y


def create_data_lstm_ed(data, seq_len, r, input_dim, output_dim, horizon):
    K = data.shape[1]
    T = data.shape[0]
    bm = binary_matrix(r, T, K)
    _data = data.copy()
    _std = np.std(data)

    _data[bm == 0] = np.random.uniform(_data[bm == 0] - _std, _data[bm == 0] + _std)

    en_x = np.zeros(shape=((T - seq_len - horizon) * K, seq_len, input_dim))
    de_x = np.zeros(shape=((T - seq_len - horizon) * K, horizon, 1))
    de_y = np.zeros(shape=((T - seq_len - horizon) * K, horizon, 1))

    for k in range(K):
        for i in range(T - seq_len - horizon):
            en_x[i] = np.expand_dims(_data[i:i + seq_len, k], axis=2)
            de_x[i] = np.expand_dims(data[i + seq_len - 1:i + seq_len + horizon - 1, k], axis=2)
            de_y[i] = np.expand_dims(data[i + seq_len:i + seq_len + horizon, k], axis=2)

    return en_x, de_x, de_y


def load_dataset_lstm_ed(seq_len, horizon, input_dim, output_dim, raw_dataset_dir, r, p, **kwargs):
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

    encoder_input_train, decoder_input_train, decoder_target_train = create_data_lstm_ed_ver_cuc_xin(train_data2d_norm,
                                                                                                     seq_len=seq_len,
                                                                                                     r=r,
                                                                                                     input_dim=input_dim,
                                                                                                     output_dim=output_dim,
                                                                                                     horizon=horizon)
    encoder_input_val, decoder_input_val, decoder_target_val = create_data_lstm_ed_ver_cuc_xin(valid_data2d_norm,
                                                                                               seq_len=seq_len, r=r,
                                                                                               input_dim=input_dim,
                                                                                               output_dim=output_dim,
                                                                                               horizon=horizon)
    encoder_input_eval, decoder_input_eval, decoder_target_eval = create_data_lstm_ed_ver_cuc_xin(test_data2d_norm,
                                                                                                  seq_len=seq_len, r=r,
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


def load_graph_data(pkl_filename):
    adj_mx = load_pickle(pkl_filename)
    return adj_mx


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


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        """

        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        if shuffle:
            permutation = np.random.permutation(self.size)
            xs, ys = xs[permutation], ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


def add_simple_summary(writer, names, values, global_step):
    """
    Writes summary for a list of scalars.
    :param writer:
    :param names:
    :param values:
    :param global_step:
    :return:
    """
    for name, value in zip(names, values):
        summary = tf.compat.v1.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        writer.add_summary(summary, global_step)


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    with np.errstate(divide='ignore', invalid='ignore'):
        adj_mx = sp.coo_matrix(adj_mx)
        d = np.array(adj_mx.sum(1))
        d_inv = np.power(d, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
        return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)


def config_logging(log_dir, log_filename='info.log', level=logging.INFO):
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Create the log directory if necessary.
    try:
        os.makedirs(log_dir)
    except OSError:
        pass
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level=level)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level=level)
    logging.basicConfig(handlers=[file_handler, console_handler], level=level)


def get_total_trainable_parameter_size():
    """
    Calculates the total number of trainable parameters in the current graph.
    :return:
    """
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        total_parameters += np.product([x.value for x in variable.get_shape()])
    return total_parameters


def create_data_dcrnn_ver_2(data, seq_len, r, input_dim, output_dim, horizon):
    K = data.shape[1]
    T = data.shape[0]
    bm = binary_matrix(r, T, K)
    _data = data.copy()
    _std = np.std(data)

    _data[bm == 0] = np.random.uniform(_data[bm == 0] - _std, _data[bm == 0] + _std)

    X = np.zeros(shape=((T - seq_len - horizon), seq_len, K, input_dim))
    Y = np.zeros(shape=((T - seq_len - horizon), horizon, K, output_dim))

    for i in range(T - seq_len - horizon):
        # X[i] = np.expand_dims(_data[i:i+seq_len], axis=2)
        X[i, :, :, 0] = _data[i:i + seq_len]
        X[i, :, :, 1] = bm[i:i + seq_len]
        Y[i, :, :, 0] = data[i + seq_len - 1:i + seq_len + horizon - 1]
    return X, Y


def create_data_dcrnn(data, seq_len, r, input_dim, output_dim, horizon):
    K = data.shape[1]
    T = data.shape[0]
    bm = binary_matrix(r, T, K)
    stdev_mx = list()
    X, Y = list(), list()
    for col in range(K):
        data_load_area = data[:, col]
        # x' - standard deviation for the training set
        x_stdev = np.std(data_load_area)
        stdev_mx.append(x_stdev)

    # convert to array
    stdev_mx_arr = np.asarray(stdev_mx).reshape(1, -1)
    stdev_mx_arr = np.insert(stdev_mx_arr, [0] * (seq_len + horizon - 1), stdev_mx_arr[0], axis=0)
    for i in range(T - seq_len - horizon):
        x = data[i:i + seq_len, :]
        y = data[i + seq_len:i + seq_len + horizon, :]
        _bm = bm[i + seq_len + horizon, :]
        _tmp_x_y = np.vstack((x, y))
        updated_x_y = stdev_mx_arr * (1.0 - _bm) + _tmp_x_y * _bm
        x = updated_x_y[:seq_len, :]
        y = updated_x_y[-horizon:, :]

        _x = np.reshape(x, x.shape + (input_dim,))
        _y = np.reshape(y, y.shape + (output_dim,))
        X.append(_x)
        Y.append(_y)
    X = np.stack(X, axis=0)
    Y = np.stack(Y, axis=0)
    return X, Y


def load_dataset_dcrnn(test_batch_size=None, **kwargs):
    batch_size = kwargs['data'].get('batch_size')
    raw_dataset_dir = kwargs['data'].get('raw_dataset_dir')
    input_dim = kwargs['model'].get('input_dim')
    output_dim = kwargs['model'].get('output_dim')
    horizon = kwargs['model'].get('horizon')
    seq_len = kwargs['model'].get('seq_len')
    r = kwargs['model'].get('verified_percentage')
    p = kwargs['data'].get('len_data')
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

    x_train, y_train = create_data_dcrnn_ver_2(train_data2d_norm, seq_len=seq_len, r=r,
                                               input_dim=input_dim, output_dim=output_dim, horizon=horizon)
    x_val, y_val = create_data_dcrnn_ver_2(valid_data2d_norm, seq_len=seq_len, r=r,
                                           input_dim=input_dim, output_dim=output_dim, horizon=horizon)
    x_eval, y_eval = create_data_dcrnn_ver_2(test_data2d_norm, seq_len=seq_len, r=r,
                                             input_dim=input_dim, output_dim=output_dim, horizon=horizon)

    for cat in ["train", "val", "eval"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        data['x_' + cat] = _x
        data['y_' + cat] = _y
        np.savez_compressed(
            os.path.join(kwargs['data'].get('output_dir'), "%s.npz" % cat),
            x=_x,
            y=_y,
        )
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, shuffle=True)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], test_batch_size, shuffle=False)
    data['eval_loader'] = DataLoader(data['x_eval'], data['y_eval'], test_batch_size, shuffle=False)
    data['scaler'] = scaler

    return data
