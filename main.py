from pandas import read_csv

from model.arima import model_arima

if __name__ == "__main__":
    # read file skip datetime column
    data_load_area = read_csv('data/data_autocorrelation.csv', usecols=range(1, 30))
    model_arima(data_load_area)
    # df = read_csv('data/data_grid.csv')
    # print(df.groupby('is_verified').size())
    # 86,62%. FALSE: 70608. TRUE: 527952
