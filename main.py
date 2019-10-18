from pandas import read_csv
from model.arima import check_stationary, model_arima

if __name__ == "__main__":
    data_load_area = read_csv('data/data_autocorrelation.csv')
    model_arima(data_load_area)