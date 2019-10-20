from pandas import read_csv
# import numpy
from model.arima import model_arima
from model.dcrnn import prepare_data
from utils.constant import LOAD_AREAS

if __name__ == "__main__":
    # read file skip datetime column
    data_load_area = read_csv('data/data_autocorrelation.csv', usecols=range(1, len(LOAD_AREAS)+1))
    x, y = prepare_data(data_load_area, p=0.3)

