from pandas import read_csv
from model.arima import model_arima, binary_matrix
from model.seq2seq_lstm import prepare_data
from sklearn import preprocessing
import numpy as np
import random

if __name__ == "__main__":
    # read file skip datetime column
    data_load_area = read_csv('data/data_autocorrelation.csv', usecols=range(1,30))
    nodes = 29
    l = 100
    r = 0.8
    h = 1
    p = 0.6
    model_arima(data_load_area, nodes, l, r, h, p)
    # df = read_csv('data/data_grid.csv')
    # print(df.groupby('is_verified').size())
    # 86,62%. FALSE: 70608. TRUE: 527952