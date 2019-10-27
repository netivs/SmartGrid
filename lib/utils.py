import logging
import numpy as np
import os
import random
import pandas as pd
import pickle
import sys
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lib.constant import LOAD_AREAS

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
    bm = binary_matrix(r, T, K)
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
    bm = np.random.choice(tf, size=(row, col), p=[r, 1.0 - r])
    return bm

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
        summary = tf.Summary()
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


def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger


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


def load_dataset_dcrnn(test_batch_size=None, **data_kwargs):
    dataset_dir = data_kwargs.get('raw_dataset_dir')
    batch_size = data_kwargs.get('batch_size')
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
        data['y_' + category][..., 0] = scaler.transform(data['y_' + category][..., 0])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, shuffle=True)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], test_batch_size, shuffle=False)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size, shuffle=False)
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