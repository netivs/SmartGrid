from pandas import read_csv
from model.arima import model_arima
from utils.constant import LOAD_AREAS

if __name__ == "__main__":
    # read file skip datetime column
    data_load_area = read_csv('data/data_load_area.csv', usecols=LOAD_AREAS)
    for i in range(10):
        print("TEST ARIMA:", i+1)
        model_arima(data_load_area)
